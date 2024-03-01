--[[
copy of my ann.lua class but using matrix.ffi instead to hopefully speed things up a bit 
but I'm only seeing a 3% increase in speed (For very small networks, like the xor problem...)
--]]
local matrix = require 'matrix.ffi'
local class = require 'ext.class'

local function initWeights(h, w)
	local m = matrix.zeros{h, w}
	--[[ maybe faster? meh
	for i=0,h*w-1 do
		m.ptr[i] = math.random() * 2 - 1
	end
	--]]
	-- [[ same ordering as ann.lua
	-- using this method tests/xor.lua produces same results for ann.lua and ann-ffi.lua
	-- tho I could just swap the ordering of ann.lua, and use the single-loop here , but meh
	for i=0,h-1 do
		for j=0,w-1 do
			m.ptr[i + h * j] = math.random() * 2 - 1
		end
	end
	--]]
	return m
end

-- c_i = a_ij b_j
local function multiplyWithBias(m, vin, vout, useBias)
	local h, w = m.size_:unpack()
	if w ~= #vin+1 then error("expected weights width "..w.." to equal input size "..#vin.." + 1") end
	assert(h == #vout)
	for i=0,h-1 do
		local s = 0
		for j=0,w-2 do
			s = s + m.ptr[i + h * j] * vin.ptr[j]
		end
		if useBias then
			s = s + m.ptr[i + h * (w-1)]
		end
		vout.ptr[i] = s
	end
end

--[[
self.x[1..n]
self.w[1..n-1]
self.useBias[1..n-1]
self.input
self.output
self.xErr[1..n]
self.netErr[1..n]
self.outputError
self.desired
--]]
local ANN = class()

local function tanhDeriv(x,y) return 1 - y * y end

ANN.activation = math.tanh
ANN.activationDeriv = tanhDeriv
-- specify activation per-layer, if not found then the default is used
ANN.perLayerActivations = {}
ANN.perLayerActivationDerivs= {}

-- false by default, 
-- set to 'true' to separate the weight-delta calculation/accumulation from the weight-delta updating the weight
ANN.useBatch = false
ANN.batchCounter = 0
ANN.totalBatchCounter = 0	-- keep track of which overall weight-accumulation we are on

function ANN:init(...)
	local layerSizes = {...}
	self.x = {}
	self.xErr = {}
	self.w = {}
	if self.useBatch then
		self.dw = {}
	end
	self.useBias = {}
	self.net = {}
	self.netErr = {}
	for i=1,#layerSizes do
		self.x[i] = matrix.zeros{layerSizes[i]}
		self.xErr[i] = matrix.zeros{layerSizes[i]}
		if i<#layerSizes then
			self.w[i] = initWeights(layerSizes[i+1], layerSizes[i]+1)
			self.useBias[i] = true
			self.net[i] = matrix.zeros{layerSizes[i+1]}
			self.netErr[i] = matrix.zeros{layerSizes[i+1]}
			if self.useBatch then
				self.dw[i] = matrix.zeros{layerSizes[i+1], layerSizes[i]+1}
			end
		end
	end
	self.input = self.x[1]
	self.output = self.x[#self.x]
	self.outputError = self.xErr[#self.xErr]
	self.desired = matrix.zeros{#self.output}
end

function ANN:feedForward()
	for i=1,#self.w do
		local activation = self.perLayerActivations[i] or self.activation
		multiplyWithBias(self.w[i], self.x[i], self.net[i], self.useBias[i])
		for j=0,#self.net[i]-1 do
			self.x[i+1].ptr[j] = activation(self.net[i].ptr[j])
		end
	end
end

--[[
values:
	
e = 1/2 sum_i (d_i - x_n,i)^2
x_n,i = f(net_n-1,i)
net_n,i = w_n,i,j * x_n,j

derivatives:

de/dx_n,j = d_j - x_n,j
dx_n,j/dnet_n-1,j = f'(net_n-1,i), all others are zero
dnet_n,i/dx_n,j = w_n,i,j
dnet_n,i/dw_n,ij = x_n,j, all others are zero
--]]
function ANN:calcError()
	assert(#self.desired == #self.outputError)
	local s = 0
	for i=0,#self.outputError-1 do
		local delta = self.desired.ptr[i] - self.output.ptr[i]
		self.outputError.ptr[i] = delta
		s = s + delta * delta
	end
	return .5 * s
end

function ANN:clearBatch()
	if not self.useBatch then return end
	for i=#self.x-1,1,-1 do
		for j=1,#self.dw[i] do
			local w = #self.x[i]
			for k=0,w-1 do
				self.dw[i].ptr[j + h * k] = 0
			end
			if self.useBias[i] then
				self.dw[i].ptr[j + h * w] = 0
			end
		end	
	end
end

function ANN:backPropagate(dt)
	dt = dt or 1
	for i=#self.x-1,1,-1 do
		local activationDeriv = self.perLayerActivationDerivs[i] or self.activationDeriv
		assert(#self.netErr[i] == #self.x[i+1])
		local w = #self.x[i]
		local h = #self.netErr[i]
		for j=0,h-1 do
			self.netErr[i].ptr[j] = self.xErr[i+1].ptr[j] * activationDeriv(self.net[i].ptr[j], self.x[i+1].ptr[j])
		end
		-- back-propagate error
		for j=0,w-1 do
			local s = 0
			for k=0,h-1 do
				s = s + self.w[i].ptr[k + h * j] * self.netErr[i].ptr[k]
			end
			self.xErr[i].ptr[j] = s
		end

		-- adjust new weights
		if not self.useBatch then
			-- ... directly/immediately
			for k=0,w-1 do
				for j=0,h-1 do
					self.w[i].ptr[j + h * k] = self.w[i].ptr[j + h * k] + dt * self.netErr[i].ptr[j] * self.x[i].ptr[k]
				end
			end
			if self.useBias[i] then
				for j=0,h-1 do
					self.w[i].ptr[j + h * w] = self.w[i].ptr[j + h * w] + dt * self.netErr[i].ptr[j]
				end
			end
		else
			-- ... accumulate into dw
			for k=0,w-1 do
				for j=0,h-1 do
					self.dw[i].ptr[j + h * k] = self.dw[i].ptr[j + h * k] + dt * self.netErr[i].ptr[j] * self.x[i].ptr[k]
				end
			end
			if self.useBias[i] then
				for j=0,h-1 do
					self.dw[i].ptr[j + h * w] = self.dw[i].ptr[j + h * w] + dt * self.netErr[i].ptr[j]
				end
			end
		end
	end
	if self.useBatch then
		self.totalBatchCounter = self.totalBatchCounter + 1
		self.batchCounter = self.batchCounter + 1
		if self.batchCounter >= self.useBatch then
			self:updateBatch()
			self.batchCounter = 0
		end
	end
end

-- update weights by batch ... and then clear the batch
function ANN:updateBatch()
	if not self.useBatch then return end
	for i=#self.x-1,1,-1 do
		local w = #self.x[i]
		local h = #self.netErr[i]
		if self.useBias[i] then
			w = w + 1
		end
		for jk=0,w*h-1 do
			self.w[i].ptr[jk] = self.w[i].ptr[jk] + self.dw[i].ptr[jk]
		end
	end
	self:clearBatch()
end

return ANN
