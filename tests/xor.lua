#!/usr/bin/env luajit

local table = require 'ext.table'
local range = require 'ext.range'
local gnuplot = require 'gnuplot'
--local ANN = require 'neuralnet.ann'
local ANN = require 'neuralnet.ann-ffi'

--math.randomseed(os.time())
math.randomseed(1)

-- a xor b == (a and not b) or (not a or b)
-- which means it'll take union of two hulls / hyperplane-collections to classify
-- which means it'll need a hidden layer and biases to combine them
-- [[
local nn = ANN(2,2,1)
--]]
--[[ logical-AND
local nn = ANN(2,1)
--]]

--[[
local ident = function(x,y) return x end
local one = function(x,y) return 1 end
nn.activation = ident
nn.activationDeriv = one
--]]
-- [[
local hi = .9
local lo = -.9
--]]
--[[
local hi = 1
local lo = -1
--]]
nn.dt = .1

local results = range(5):map(function() return table() end)

-- first test: teach it and/or 
for i=1,100000 do
	local a = math.random(2) == 2
	local b = math.random(2) == 2
	-- [[ xor	
	local c = a ~= b	--xor problem.  requires two layers to solve:  (a & ~b) | (~a & b) ... or ... (a | b) & (~a | ~b) ... either is 2 operations deep <=> two layers
	--]]
	--[[ and
	local c = a and b
	--]]
	nn.input[1] = a and hi or lo
	nn.input[2] = b and hi or lo
	--print('input', nn.input)
	--print('input weight', nn.w[1])
	--print('mid weight', nn.w[2])
	nn:feedForward()
	--print('net mid', nn.net[1])
	--print('mid', nn.x[2])
	--print('net out', nn.net[2])
	--print('out', nn.x[3])
	nn.desired[1] = c and hi or lo
	--print('desired', nn.desired)
	local err = nn:calcError()
	results[1]:insert(nn.input[1])
	results[2]:insert(nn.input[2])
	results[3]:insert(nn.output[1])
	results[4]:insert(nn.desired[1])
	results[5]:insert(err)
	
	--print('de/dy', nn.xErr[3])
	nn:backPropagate()
	--print('de/dnet y', nn.netErr[2])
	--print('de/dmid', nn.xErr[2])
	--print('de/dnet mid', nn.netErr[1])
	--print('de/dx', nn.xErr[1])
	--print()
end

for i,w in ipairs(nn.w) do
	print('layer '..i..':')
	print(w)
end

gnuplot{
	terminal = 'png size 1024,768',
	output = 'xor.png',
	style = 'data lines',
	log = 'y',
	data = results,
	{using='0:5', title='error'},
}
