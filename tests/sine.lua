#! /usr/bin/env luajit
--[[
ok so after complex_mul's sucess, now to try to piecewise approximate a sine wave
--]]
local table = require 'ext.table'
local ANN = require 'neuralnet.ann'
nn = ANN(1,10,1)

local f = math.sin
local xmin = 0
local xmax = 2 * math.pi

local results = table()
-- right at 100,000 the max error seem to drop
for i=1,1000000 do
	local t = math.random() * (xmax - xmin) + xmin
	nn.input[1] = t
	nn:feedForward()
	nn.desired[1] = f(t)
	local err = nn:calcError()
	results:insert(err)
	nn:backPropagate(.1)
end

require 'gnuplot'{
	terminal = 'png size 1024,768',
	output = 'sine.png',
	style = 'data lines',
	log = 'xy',
	data = {results},
	{using='0:1', title='error'},
}

-- final result ...
local xs = table()
local ys = table()
local des = table()
local n = 1000
for i=1,n do
	local x = (i-.5)/n * (xmax - xmin) + xmin
	nn.input[1] = x
	nn:feedForward()
	local y = nn.output[1]
	xs:insert(x)
	ys:insert(y)
	des:insert(f(x))
end

require 'gnuplot'{
	terminal = 'png size 1024,768',
	output = 'sine_result.png',
	style = 'data lines',
	data = {xs, ys, des},
	{using='1:2', title='approximation'},
	{using='1:3', title='correct'},
}


