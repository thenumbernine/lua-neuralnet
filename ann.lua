local matrix = require 'matrix'
--local matrix = require 'matrix.ffi' -- still segfaults, still not a perfect replacement
local class = require 'ext.class'
local assertindex = require 'ext.assert'.index
local activations = require 'neuralnet.activation'

--[[
self.x[1..n]
self.net[1..n-1]
self.w[1..n-1]
self.useBias[1..n-1]
self.activations[1..n-1]
self.activationDerivs[1..n-1]
self.input
self.output
self.xErr[1..n]
self.netErr[1..n]
self.outputError
self.desired
--]]
local ANN = class()

function ANN:newMatrix(...)
	return matrix.zeros(...)
end

function ANN:newWeights(h, w)
	local m = self:newMatrix(h, w)
	for i=1,h do
		for j=1,w do
			m[i][j] = math.random() * 2 - 1
		end
	end
	return m
end

-- c_i = a_ij b_j
local function multiplyWithBias(m, vin, vout, useBias)
	local h = #m
	local w = #m[1]
	if w ~= #vin+1 then error("expected weights width "..w.." to equal input size "..#vin.." + 1") end
	if h ~= #vout then error("expected weights height "..h.." to equal output size "..#vout) end
	for i=1,h do
		local s = 0
		for j=1,w-1 do
			s = s + m[i][j] * vin[j]
		end
		if useBias then
			s = s + m[i][w]
		end
		vout[i] = s
	end
end

-- false by default,
-- set to 'true' to separate the weight-delta calculation/accumulation from the weight-delta updating the weight
ANN.useBatch = false
ANN.batchCounter = 0	-- keep track of which weight-accumulation we are on

-- what % of matrix *COLUMNS* (i.e. inputs) to zero
-- seems I've seen some articles say 'zero rows', while others show equations of zeroed lhs matrices, which is zeroing columns, so :shrug:
-- zeroing the output doesn't seem practical, so zeroing the last layer rows doesn't seem practical
-- NOTICE - this is exclusive with dilution.  if this is <1 then it will be used first.
ANN.dropout = 1

-- how many % weights to keep per update
ANN.dilution = 1

function ANN:init(...)
	local layerSizes = {...}
	self.x = {}
	self.xErr = {}
	self.w = {}
	self.dw = {}	-- for useBatch
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
			self.dw[i] = self:newMatrix(layerSizes[i+1], layerSizes[i]+1)	-- for useBatch
		end
	end
	self.input = self.x[1]
	self.output = self.x[#self.x]
	self.outputError = self.xErr[#self.xErr]
	self.desired = self:newMatrix(#self.output)

	self.activations = {}
	self.activationDerivs = {}
	self:setActivation'tanh'
	self:setActivationDeriv'tanhDeriv'
end

function ANN:setActivation(func, index)
	if type(func) == 'string' then func = assertindex(activations.funcs, func) end
	if index then
		self.activations[index] = func
	else
		for i=1,#self.w do
			self.activations[i] = func
		end
	end
end

function ANN:setActivationDeriv(func, index)
	if type(func) == 'string' then func = assertindex(activations.funcs, func) end
	if index then
		self.activationDerivs[index] = func
	else
		for i=1,#self.w do
			self.activationDerivs[i] = func
		end
	end
end


function ANN:feedForward()
	for k=1,#self.w do
		--[[
		multiplyWithBias(self.w[k], self.x[k], self.net[k], self.useBias[k])
		--]]
		-- [[ inline is usu faster
		local m = self.w[k]
		local vin = self.x[k]
		local vout = self.net[k]
		local useBias = self.useBias[k]
		local h = #m
		local w = #m[1]
		if w ~= #vin+1 then error("expected weights width "..w.." to equal input size "..#vin.." + 1") end
		if h ~= #vout then error("expected weights height "..h.." to equal output size "..#vout) end
		for i=1,h do
			local s = 0
			for j=1,w-1 do
				s = s + m[i][j] * vin[j]
			end
			if useBias then
				s = s + m[i][w]
			end
			vout[i] = s
		end
		--]]

		local activation = self.activations[k]
		for i=1,#self.net[k] do
			self.x[k+1][i] = activation(self.net[k][i])
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
	for i=1,#self.outputError do
		local delta = self.desired[i] - self.output[i]
		self.outputError[i] = delta
		s = s + delta * delta
	end
	return .5 * s
end

function ANN:clearBatch()
	if not self.useBatch then return end
	for i=#self.x-1,1,-1 do
		for j=1,#self.dw[i] do
			local l = #self.x[i]
			for k=1,l do
				self.dw[i][j][k] = 0
			end
			if self.useBias[i] then
				self.dw[i][j][l+1] = 0
			end
		end
	end
end

ANN.dt = 1
function ANN:backPropagate(dt)
	dt = dt or self.dt
	for i=#self.x-1,1,-1 do
		local activationDeriv = self.activationDerivs[i]
		assert(#self.netErr[i] == #self.x[i+1])
		for j=1,#self.x[i+1] do
			self.netErr[i][j] = self.xErr[i+1][j] * activationDeriv(self.net[i][j], self.x[i+1][j])
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
		if not self.useBatch then
			-- ... directly/immediately
			if self.dropout == 1 and self.dilution == 1 then
				for j=1,#self.w[i] do
					local l = #self.x[i]
					for k=1,l do
						self.w[i][j][k] = self.w[i][j][k] + dt * self.netErr[i][j] * self.x[i][k]
					end
					if self.useBias[i] then
						self.w[i][j][l+1] = self.w[i][j][l+1] + dt * self.netErr[i][j]
					end
				end
			elseif self.dropout < 1 then
				local l = #self.x[i]
				for k=1,l do
					if math.random() < self.dropout then
						for j=1,#self.w[i] do
							self.w[i][j][k] = self.w[i][j][k] + dt * self.netErr[i][j] * self.x[i][k]
						end
					end
				end
				if self.useBias[i] then
					if math.random() < self.dropout then
						for j=1,#self.w[i] do
							self.w[i][j][l+1] = self.w[i][j][l+1] + dt * self.netErr[i][j]
						end
					end
				end
			else	-- dilution
				for j=1,#self.w[i] do
					local l = #self.x[i]
					for k=1,l do
						if math.random() < self.dilution then
							self.w[i][j][k] = self.w[i][j][k] + dt * self.netErr[i][j] * self.x[i][k]
						end
					end
					if self.useBias[i] then
						if math.random() < self.dilution then
							self.w[i][j][l+1] = self.w[i][j][l+1] + dt * self.netErr[i][j]
						end
					end
				end
			end
		else
			-- ... accumulate into dw
			-- should you apply dilution before or after batch weight accum?
			-- I vote after, because doing so before just screws up our gradient direction, and the whole point of batch is to better determine the gradient direction
			for j=1,#self.dw[i] do
				local l = #self.x[i]
				for k=1,l do
					self.dw[i][j][k] = self.dw[i][j][k] + dt * self.netErr[i][j] * self.x[i][k]
				end
				if self.useBias[i] then
					self.dw[i][j][l+1] = self.dw[i][j][l+1] + dt * self.netErr[i][j]
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
	if self.dropout == 1 and self.dilution == 1 then
		for i=#self.x-1,1,-1 do
			for j=1,#self.w[i] do
				local l = #self.x[i]
				for k=1,l do
					self.w[i][j][k] = self.w[i][j][k] + self.dw[i][j][k]
				end
				if self.useBias[i] then
					self.w[i][j][l+1] = self.w[i][j][l+1] + self.dw[i][j][l+1]
				end
			end
		end
	elseif self.dropout < 1 then
		for i=#self.x-1,1,-1 do
			local l = #self.x[i]
			for k=1,l do
				if math.random() < self.dropout then
					for j=1,#self.w[i] do
						self.w[i][j][k] = self.w[i][j][k] + self.dw[i][j][k]
					end
				end
			end
			if self.useBias[i] then
				if math.random() < self.dropout then
					for j=1,#self.w[i] do
						self.w[i][j][l+1] = self.w[i][j][l+1] + self.dw[i][j][l+1]
					end
				end
			end
		end
	else	-- dilution
		for i=#self.x-1,1,-1 do
			local l = #self.x[i]
			for j=1,#self.w[i] do
				for k=1,l do
					if math.random() < self.dilution then
						self.w[i][j][k] = self.w[i][j][k] + self.dw[i][j][k]
					end
				end
				if self.useBias[i] then
					if math.random() < self.dilution then
						self.w[i][j][l+1] = self.w[i][j][l+1] + self.dw[i][j][l+1]
					end
				end
			end
		end
	end
	self:clearBatch()
end

return ANN
