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


local function showCurrentResult(args)
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

	require 'gnuplot'(table({
		terminal = 'svg size 1024,768',
		output = 'output.svg',
		style = 'data lines',
		data = {xs, ys, des},
		{using='1:2', title='approximation'},
		{using='1:3', title='correct'},
	}, args))
end



local results = table()
-- right at 100,000 the max error seem to drop to .01.  then at 10,000,000 it drops to .0001
local lastLog10
for i=1,10000000 do
	local t = math.random() * (xmax - xmin) + xmin
	nn.input[1] = t
	nn:feedForward()
	nn.desired[1] = f(t)
	local err = nn:calcError()
	results:insert(err)
	nn:backPropagate(.1)

	local log10 = math.floor(math.log(i) / math.log(10))
	if log10 ~= lastLog10 then
		lastLog10 = log10
		showCurrentResult{
			output = 'sine_result_'..('%.1e'):format(i)..'.svg',
			title = 'results after '..('%.1e'):format(i)..' iterations',
		}
	end
end

require 'gnuplot'{
	terminal = 'svg size 1024,768',
	output = 'sine_error.svg',
	style = 'data points',
	log = 'xy',
	data = {results},
	{using='0:1', title='error'},
}

showCurrentResult{output='sine_result_final.svg'}

