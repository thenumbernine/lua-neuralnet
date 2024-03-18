--[[
copy of my ann.lua class but using matrix.ffi instead to hopefully speed things up a bit
but I'm only seeing a 3% increase in speed (For very small networks, like the xor problem...)
--]]
local ffi = require 'ffi'
local matrix = require 'matrix.ffi'
local class = require 'ext.class'
local asserteq = require 'ext.assert'.eq
local assertne = require 'ext.assert'.ne

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

local rowmajor = true

-- since just getmetatable(nn.w) isn't enough to preserve rowmajor etc
function ANN:newMatrix(...)
	local m = matrix.zeros({...}, nil, rowmajor)
	asserteq(m.rowmajor, rowmajor)
	return m
end

function ANN:newWeights(h, w)
	local m = self:newMatrix(h, w)
	local mptr = m.ptr
	for i=0,h-1 do
		for j=0,w-1 do
			if rowmajor then
				mptr[i * w + j] = math.random() * 2 - 1
			else
				mptr[i + h * j] = math.random() * 2 - 1
			end
		end
	end
	return m
end

-- c_i = a_ij b_j
local function multiplyWithBias(m, vin, vout, useBias)
	local mptr = m.ptr
	local vinptr = vin.ptr
	local voutptr = vout.ptr
	local h, w = m.size_:unpack()
	asserteq(w, #vin + 1)
	asserteq(h, #vout)
	for i=0,h-1 do
		local s = 0
		for j=0,w-2 do
			s = s + mptr[i * w + j] * vinptr[j]
		end
		if useBias then
			s = s + mptr[i * w + (w-1)]
		end
		voutptr[i] = s
	end
end

ANN.activation, ANN.activationDeriv = table.unpack(require 'neuralnet.activation'.tanh)

-- specify activation per-layer, if not found then the default is used
ANN.perLayerActivations = {}
ANN.perLayerActivationDerivs= {}

-- false by default,
-- set to 'true' to separate the weight-delta calculation/accumulation from the weight-delta updating the weight
ANN.useBatch = false
ANN.batchCounter = 0	-- keep track of which overall weight-accumulation we are on

function ANN:init(...)
	local layerSizes = {...}
	self.x = {}
	self.xErr = {}
	self.w = {}
	self.dw = {}
	self.useBias = {}
	self.net = {}
	self.netErr = {}
	for i=1,#layerSizes do
		self.x[i] = self:newMatrix(layerSizes[i])
		self.xErr[i] = self:newMatrix(layerSizes[i])
		if i<#layerSizes then
			self.w[i] = self:newWeights(layerSizes[i+1], layerSizes[i]+1)
			self.useBias[i] = true
			self.net[i] = self:newMatrix(layerSizes[i+1])
			self.netErr[i] = self:newMatrix(layerSizes[i+1])
			self.dw[i] = self:newMatrix(layerSizes[i+1], layerSizes[i]+1)
		end
	end
	self.input = self.x[1]
	self.output = self.x[#self.x]
	self.outputError = self.xErr[#self.xErr]
	self.desired = self:newMatrix(#self.output)
end

function ANN:feedForward()
	for k=1,#self.w do
		--[[
		multiplyWithBias(self.w[k], self.x[k], self.net[k], self.useBias[k])
		--]]
		-- [[ inline
		local xk = self.x[k]
		local netk = self.net[k]
		local useBias = self.useBias[k]
		local mptr = self.w[k].ptr
		local xkptr = xk.ptr
		local netkptr = netk.ptr
		local h, w = self.w[k].size_:unpack()
--DEBUG: asserteq(w, #xk + 1)
--DEBUG: asserteq(h, #netk)
		local activation = self.perLayerActivations[k] or self.activation
		for i=0,h-1 do
			local s = 0
			for j=0,w-2 do
				s = s + mptr[i * w + j] * xkptr[j]
			end
			if useBias then
				s = s + mptr[i * w + (w-1)]
			end
			netkptr[i] = s
		end
		local netptr = self.net[k].ptr
		local xptr = self.x[k+1].ptr
		for i=0,#self.net[k]-1 do
			xptr[i] = activation(netptr[i])
		end	
		--]]
		--[[ inline more ... the more I inline / make native the operations, the slower luajit goes ...
		local m = self.w[k]
		local useBias = self.useBias[k]
		local activation = self.perLayerActivations[k] or self.activation
		local mijptr = m.ptr
--DEBUG: asserteq(m.size_[2], #self.x[k] + 1)
--DEBUG: asserteq(m.size_[1], #self.net[k])
		local xptr = self.x[k].ptr
		local xendptr = xptr + m.size_[2] - 1	-- stop short of the bias
		local netiptr = self.net[k].ptr
		local netiendptr = netiptr + m.size_[1]
		local yiptr = self.x[k+1].ptr
		while netiptr < netiendptr  do
			local xjptr = xptr
			netiptr[0] = mijptr[0] * xjptr[0]
			mijptr = mijptr + 1ULL
			xjptr = xjptr + 1ULL
			while xjptr < xendptr do
				netiptr[0] = netiptr[0] + mijptr[0] * xjptr[0]
				mijptr = mijptr + 1ULL
				xjptr = xjptr + 1ULL
			end
			if useBias then
				netiptr[0] = netiptr[0] + mijptr[0]
			end
			mijptr = mijptr + 1ULL
			yiptr[0] = activation(netiptr[0])
			netiptr = netiptr + 1ULL
			yiptr = yiptr + 1ULL
		end
--DEBUG: asserteq(mijptr, m.ptr + m.size_[2] * m.size_[1])
--DEBUG: asserteq(netiptr, self.net[k].ptr + m.size_[1])
--DEBUG: asserteq(yiptr, self.x[k+1].ptr + m.size_[1])
		--]]

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
--DEBUG: asserteq(#self.desired, #self.outputError)
	local desiredptr = self.desired.ptr
	local outputptr = self.output.ptr
	local outputErrorptr = self.outputError.ptr
	local s = 0
	for i=0,#self.outputError-1 do
		local delta = desiredptr[i] - outputptr[i]
		outputErrorptr[i] = delta
		s = s + delta * delta
	end
	return .5 * s
end

function ANN:clearBatch()
	if not self.useBatch then return end
	for i=#self.x-1,1,-1 do
		local dw = self.dw[i]
		local h, w = table.unpack(dw.size_)
--DEBUG: asserteq(w, #self.x[i] + 1)
		ffi.fill(dw.ptr, ffi.sizeof(dw.ctype) * w * h)
	end
end

ANN.dt = 1
function ANN:backPropagate(dt)
	dt = dt or self.dt
	for i=#self.x-1,1,-1 do
		local weight = self.w[i]
		local wptr = weight.ptr
		local h, w = table.unpack(weight.size_)
		local dweight
		local dwptr
		if self.dw then
			dweight = self.dw[i]
			dwptr = dweight.ptr
		end
		local activationDeriv = self.perLayerActivationDerivs[i] or self.activationDeriv
		local nextxptr = self.x[i+1].ptr
		local xptr = self.x[i].ptr
		local netptr = self.net[i].ptr
		local netErr = self.netErr[i]
--DEBUG: asserteq(#netErr, #self.x[i+1])
		local netErrptr = netErr.ptr
		--local w = #self.x[i]
		--local h = #netErr
		local nextxErrptr = self.xErr[i+1].ptr
		for j=0,h-1 do
			netErrptr[j] = nextxErrptr[j] * activationDeriv(netptr[j], nextxptr[j])
		end
		-- back-propagate error
		local xErrptr = self.xErr[i].ptr
		for j=0,w-2 do
			local s = 0
			for k=0,h-1 do
				s = s + wptr[k * w + j] * netErrptr[k]
			end
			xErrptr[j] = s
		end

		-- adjust new weights
		if not self.useBatch then
			-- ... directly/immediately
			for k=0,w-2 do
				for j=0,h-1 do
					wptr[j * w + k] = wptr[j * w + k] + dt * netErrptr[j] * xptr[k]
				end
			end
			if self.useBias[i] then
				for j=0,h-1 do
					wptr[j * w + (w-1)] = wptr[j * w + (w-1)] + dt * netErrptr[j]
				end
			end
		else
			-- ... accumulate into dw
			for k=0,w-2 do
				for j=0,h-1 do
					dwptr[j * w + k] = dwptr[j * w + k] + dt * netErrptr[j] * xptr[k]
				end
			end
			if self.useBias[i] then
				for j=0,h-1 do
					dwptr[j * w + (w-1)] = dwptr[j * w + (w-1)] + dt * netErrptr[j]
				end
			end
		end
	end
	if self.useBatch then
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
		local weight = self.w[i]
		local wptr = weight.ptr
		local dweight = self.dw[i]
		local dwptr = dweight.ptr
		local h, w = table.unpack(weight.size_)
		for jk=0,w*h-1 do
			wptr[jk] = wptr[jk] + dwptr[jk]
		end
	end
	self:clearBatch()
end

return ANN
