#!/usr/bin/env luajit

require 'ext'
gnuplot = require 'gnuplot'
local ANN = require 'neuralnet.ann'

math.randomseed(os.time())

local nn = ANN(2,2,1)

local results = range(5):map(function() return table() end)

-- first test: teach it and/or 
for i=1,10000 do
	local a = math.random(2) == 2
	local b = math.random(2) == 2
	local c = a ~= b	--xor problem.  requires two layers to solve:  (a & ~b) | (~a & b) ... or ... (a | b) & (~a | ~b) ... either is 2 operations deep <=> two layers
	nn.input[1] = a and 1 or -1
	nn.input[2] = b and 1 or -1
	--print('input', nn.input)
	--print('input weight', nn.w[1])
	--print('mid weight', nn.w[2])
	nn:feedForward()
	--print('net mid', nn.net[1])
	--print('mid', nn.x[2])
	--print('net out', nn.net[2])
	--print('out', nn.x[3])
	nn.desired[1] = c and 1 or -1
	--print('desired', nn.desired)
	local err = nn:calcError()
	results[1]:insert(nn.input[1])
	results[2]:insert(nn.input[2])
	results[3]:insert(nn.output[1])
	results[4]:insert(nn.desired[1])
	results[5]:insert(err)
	
	--print('de/dy', nn.xErr[3])
	nn:backPropagate(.1)
	--print('de/dnet y', nn.netErr[2])
	--print('de/dmid', nn.xErr[2])
	--print('de/dnet mid', nn.netErr[1])
	--print('de/dx', nn.xErr[1])
	--print()
end

gnuplot{
	terminal = 'png size 1024,768',
	output = 'xor.png',
	style = 'data lines',
	log = 'xy',
	data = results,
	{using='0:5', title='error'},
}
