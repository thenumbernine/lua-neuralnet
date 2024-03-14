#!/usr/bin/env luajit
--[[
performance test between feedForward and backprop, with and without batch, matrix vs matrix.ffi vs maybe I'll write my own wrappers to Blaze or something

Press ENTER or type command to continue

ann...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (2.7898669242859s)
 feedForward only...
 ...done feedForward only (0.52732181549072s)
...done ann (3.3183100223541s)

-- USING V1 ... least optimization is better ....
ann-ffi...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.2917938232422s)
 feedForward only...
 ...done feedForward only (0.38366603851318s)
...done ann-ffi (1.6803929805756s)

-- USING V2
ann-ffi...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.5053470134735s)
 feedForward only...
 ...done feedForward only (0.59355282783508s)
...done ann-ffi (2.1031799316406s)

-- USING V3
ann-ffi...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.4636361598969s)
 feedForward only...
 ...done feedForward only (0.57114696502686s)
...done ann-ffi (2.0391969680786s)

-- USING V4
ann-ffi...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.5506801605225s)
 feedForward only...
 ...done feedForward only (0.66539001464844s)
...done ann-ffi (2.2207000255585s)

so ann-ffi is only beneficial for larger networks
and any attempt to 'optimize' it makes it go slower
luajit is weird
--]]
local timer = require 'ext.timer'.timer
local numIter = 10000

-- [[
for _,info in ipairs{
	{name='ann', cl=require 'neuralnet.ann'},
	{name='ann-ffi', cl=require 'neuralnet.ann-ffi'},
} do
	print()
	timer(info.name,function()
		local nn = info.cl(222, 80, 40, 2)
		-- TODO how much is filling the input with __index vs ptr access changing things?
		-- welp filling is 1/100th the time of feedForward and backProp so :shrug:
		for i=1,#nn.input do
			nn.input[i] = math.random()
		end
		timer('feedForward + backPropagate', function()
			for i=1,numIter do
				nn:feedForward()
				nn.desired[1] = math.random()
				nn.desired[2] = math.random()
				nn:calcError()
				nn:backPropagate()
			end
		end)
		timer('feedForward only', function()
			for i=1,numIter do
				nn:feedForward()
			end
		end)
	end)
end
--]]

-- [[ the C++ version has a slightly dif API since it groups the layer stuff into sub-objects.
-- I didn't do this in Lua because more tables = more memory and more dereferences
-- but in C++ this isn't the case
for _,info in ipairs{
	{name='NeuralNetLua', cl=require 'NeuralNetLua'},
} do
	print()
	timer(info.name,function()
		local nn = info.cl(222, 80, 40, 2)
		--local input = nn:input()	-- TODO .. or I could just nn.input = nn.layers[1].x ... but I've overridden the metatable to not allow new fields to be added (why do I do that? hmm...)
		nn.input = nn.layers[1].x
		
		-- the rest matches the test above
		for i=1,#nn.input do
			nn.input[i] = math.random()
		end
		timer('feedForward + backPropagate', function()
			for i=1,numIter do
				nn:feedForward()
				nn.desired[1] = math.random()
				nn.desired[2] = math.random()
				nn:calcError()
				nn:backPropagate()
			end
		end)
		timer('feedForward only', function()
			for i=1,numIter do
				nn:feedForward()
			end
		end)
	end)
end
--]]
