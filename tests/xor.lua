#!/usr/bin/env luajit

local ANN = require 'neuralnet.ann'
local TDNN = require 'neuralnet.tdnn'

math.randomseed(os.time())

local nn = ANN(2,2,1)

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
	print(nn.input[1], nn.input[2], nn.output[1] > 0 and 1 or -1, nn.desired[1], math.log(err))
	--print('de/dy', nn.xErr[3])
	nn:backPropagate(.01)
	--print('de/dnet y', nn.netErr[2])
	--print('de/dmid', nn.xErr[2])
	--print('de/dnet mid', nn.netErr[1])
	--print('de/dx', nn.xErr[1])
	--print()
end

