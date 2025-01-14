--[[
copy of my ann.lua class but using matrix.ffi instead to hopefully speed things up a bit
but I'm only seeing a 3% increase in speed (For very small networks, like the xor problem...)
--]]
local ffi = require 'ffi'
local matrix = require 'matrix.ffi'
local class = require 'ext.class'
local assert = require 'ext.assert'
local activations = require 'neuralnet.activation'

--[[
self.x[1..n]
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

local rowmajor = true

-- since just getmetatable(nn.w) isn't enough to preserve rowmajor etc
function ANN:newMatrix(...)
	local m = matrix.zeros({...}, nil, rowmajor)
	assert.eq(m.rowmajor, rowmajor)
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
	assert.eq(w, #vin + 1)
	assert.eq(h, #vout)
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

-- false by default,
-- set to 'true' to separate the weight-delta calculation/accumulation from the weight-delta updating the weight
ANN.useBatch = false
ANN.batchCounter = 0	-- keep track of which overall weight-accumulation we are on

-- what % of matrix *COLUMNS* (i.e. inputs) to zero
-- seems I've seen some articles say 'zero rows', while others show equations of zeroed lhs matrices, which is zeroing columns, so :shrug:
-- zeroing the output doesn't seem practical, so zeroing the last layer rows doesn't seem practical
-- NOTICE - this is exclusive with dilution.  if this is <1 then it will be used first.
ANN.dropout = 1

-- how many % weights to keep per update
ANN.dilution = 1

function ANN:init(...)
	local layerSizes = {...}
	-- TODO make a layers[] table already
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

	self.activations = {}
	self.activationDerivs = {}
	self:setActivation'tanh'
	self:setActivationDeriv'tanhDeriv'
end

function ANN:setActivation(func, index)
	if type(func) == 'string' then func = assert.index(activations.funcs, func) end
	if index then
		self.activations[index] = func
	else
		for i=1,#self.w do
			self.activations[i] = func
		end
	end
end

function ANN:setActivationDeriv(func, index)
	if type(func) == 'string' then func = assert.index(activations.funcs, func) end
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
		-- [[ inline
		local xk = self.x[k]
		local netk = self.net[k]
		local useBias = self.useBias[k]
		local mptr = self.w[k].ptr
		local xkptr = xk.ptr
		local netkptr = netk.ptr
		local h, w = self.w[k].size_:unpack()
--DEBUG: assert.eq(w, #xk + 1)
--DEBUG: assert.eq(h, #netk)
		local activation = self.activations[k]
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
		local activation = self.activations[k]
		local mijptr = m.ptr
--DEBUG: assert.eq(m.size_[2], #self.x[k] + 1)
--DEBUG: assert.eq(m.size_[1], #self.net[k])
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
--DEBUG: assert.eq(mijptr, m.ptr + m.size_[2] * m.size_[1])
--DEBUG: assert.eq(netiptr, self.net[k].ptr + m.size_[1])
--DEBUG: assert.eq(yiptr, self.x[k+1].ptr + m.size_[1])
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
--DEBUG: assert.eq(#self.desired, #self.outputError)
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
--DEBUG: assert.eq(w, #self.x[i] + 1)
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
		local activationDeriv = self.activationDerivs[i]
		local nextxptr = self.x[i+1].ptr
		local xptr = self.x[i].ptr
		local netptr = self.net[i].ptr
		local netErr = self.netErr[i]
--DEBUG: assert.eq(#netErr, #self.x[i+1])
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
			if self.dropout == 1 and self.dilution == 1 then
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
			elseif self.dropout < 1 then
				for k=0,w-2 do
					if math.random() < self.dropout then
						for j=0,h-1 do
							wptr[j * w + k] = wptr[j * w + k] + dt * netErrptr[j] * xptr[k]
						end
					end
				end
				if self.useBias[i] then
					if math.random() < self.dropout then
						for j=0,h-1 do
							wptr[j * w + (w-1)] = wptr[j * w + (w-1)] + dt * netErrptr[j]
						end
					end
				end
			else	-- dilution
				for k=0,w-2 do
					for j=0,h-1 do
						if math.random() < self.dilution then
							wptr[j * w + k] = wptr[j * w + k] + dt * netErrptr[j] * xptr[k]
						end
					end
				end
				if self.useBias[i] then
					for j=0,h-1 do
						if math.random() < self.dilution then
							wptr[j * w + (w-1)] = wptr[j * w + (w-1)] + dt * netErrptr[j]
						end
					end
				end
			end
		else
			-- ... accumulate into dw
			-- should you apply dilution before or after batch weight accum?
			-- I vote after, because doing so before just screws up our gradient direction, and the whole point of batch is to better determine the gradient direction
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
	if self.dropout == 1 and self.dilution == 1 then
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
	elseif self.dropout < 1 then
		for i=#self.x-1,1,-1 do
			local weight = self.w[i]
			local wptr = weight.ptr
			local dweight = self.dw[i]
			local dwptr = dweight.ptr
			local h, w = table.unpack(weight.size_)
			for k=0,w-2 do
				if math.random() < self.dropout then
					for j=0,h-1 do
						wptr[j * w + k] = wptr[j * w + k] + dwptr[j * w + k]
					end
				end
			end
			if self.useBias[i] then
				if math.random() < self.dropout then
					for j=0,h-1 do
						wptr[j * w + (w-1)] = wptr[j * w + (w-1)] + dwptr[j * w + (w-1)]
					end
				end
			end
		end
	else	-- dilution
		for i=#self.x-1,1,-1 do
			local weight = self.w[i]
			local wptr = weight.ptr
			local dweight = self.dw[i]
			local dwptr = dweight.ptr
			local h, w = table.unpack(weight.size_)
			for jk=0,w*h-1 do
				if math.random() < self.dilution then
					wptr[jk] = wptr[jk] + dwptr[jk]
				end
			end
		end
	end
	self:clearBatch()
end

return ANN
