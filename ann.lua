require 'ext'

local Vector = class()

function Vector:init(h)
	for i=1,h do
		self[i] = 0
	end
end

function Vector:apply(src, f)
	assert(#self == #src)
	for i=1,#self do
		self[i] = f(src[i])
	end
end

function Vector:__tostring()
	return '['..table.concat(self, ',')..']'
end

function Vector.__concat(a,b) return tostring(a) .. tostring(b) end

local Matrix = class()

function Matrix:init(h,w)
	self.size = {h,w}	-- math index
	for i=1,h do
		self[i] = {}
		for j=1,w do
			self[i][j] = (math.random()-.5)*.2
		end
	end
end

-- c_i = a_ij b_j
function Matrix:multiplyWithBias(vin,vout, useBias)
	assert(self.size[2] == #vin+1)
	assert(self.size[1] == #vout)
	for i=1,self.size[1] do
		local s = 0
		for j=1,self.size[2]-1 do
			s = s + self[i][j] * vin[j]
		end
		if useBias then
			s = s + self[i][self.size[2]]
		end
		vout[i] = s
	end
end

function Matrix:__tostring()
	return '['..table.map(self, function(c,i)
		if type(i) == 'number' then
			return '['..table.concat(c, ',')..']'
		end
	end):concat(',')..']'
end

function Matrix.__concat(a,b) return tostring(a) .. tostring(b) end

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

function ANN:init(...)
	local layerSizes = {...}
	self.x = {}
	self.xErr = {}
	self.w = {}
	self.useBias = {}
	self.net = {}
	self.netErr = {}
	for i=1,#layerSizes do
		self.x[i] = Vector(layerSizes[i])
		self.xErr[i] = Vector(layerSizes[i])
		if i<#layerSizes then
			self.w[i] = Matrix(layerSizes[i+1], layerSizes[i]+1)
			self.useBias[i] = true
			self.net[i] = Vector(layerSizes[i+1])
			self.netErr[i] = Vector(layerSizes[i+1])
		end
	end
	self.input = self.x[1]
	self.output = self.x[#self.x]
	self.outputError = self.xErr[#self.xErr]
	self.desired = Vector(#self.output)
end

function ANN:feedForward()
	for i=1,#self.w do
		self.w[i]:multiplyWithBias(self.x[i], self.net[i], self.useBias[i])
		self.x[i+1]:apply(self.net[i], self.activation)
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
	for i=1,#self.outputError do
		local delta = self.desired[i] - self.output[i]
		self.outputError[i] = delta
		s = s + delta * delta
	end
	return .5 * s
end

function ANN:backPropagate(dt)
	dt = dt or 1
	for i=#self.x-1,1,-1 do
		assert(#self.netErr[i] == #self.x[i+1])
		for j=1,#self.x[i+1] do
			self.netErr[i][j] = self.xErr[i+1][j] * self.activationDeriv(self.net[i][j], self.x[i+1][j])
		end
		-- back-propagate error
		for j=1,#self.xErr[i] do
			local s = 0
			for k=1,#self.netErr[i] do
				s = s + self.w[i][k][j] * self.netErr[i][k]
			end
			self.xErr[i][j] = s
		end
		-- adjust new weights
		for j=1,self.w[i].size[1] do
			local l = #self.x[i]
			for k=1,l do
				self.w[i][j][k] = self.w[i][j][k] + dt * self.netErr[i][j] * self.x[i][k]
			end
			if self.useBias[i] then
				self.w[i][j][l+1] = self.w[i][j][l+1] + dt * self.netErr[i][j]
			end
		end
	end
end

return ANN


