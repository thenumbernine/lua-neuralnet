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

ann-ffi[double]...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.2917938232422s)
 feedForward only...
 ...done feedForward only (0.38366603851318s)
...done ann-ffi (1.6803929805756s)

ann-ffi[float]...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.5866541862488s)
 feedForward only...
 ...done feedForward only (0.4066641330719s)
...done ann-ffi (1.9986720085144s)

NeuralNet::ANN<float>...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (0.41166496276855s)
 feedForward only...
 ...done feedForward only (0.21063303947449s)
...done NeuralNetLua (0.62612009048462s)

NeuralNet::ANN<double>...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (0.48642611503601s)
 feedForward only...
 ...done feedForward only (0.23366689682007s)
...done NeuralNetLua (0.72286415100098s)

.. from there the more i try to optimize ann-ffi the slower it goes
so ann-ffi is only beneficial for larger networks
and any attempt to 'optimize' it makes it go slower
luajit is weird
--]]
local timer = require 'ext.timer'.timer
local numIter = 10000

-- hmm ann-ffi isn't much faster as float
-- maybe even 5% slower for some reason
require 'matrix.ffi'.real = 'float'

for _,info in ipairs{
	--{name='ann', ctor=require 'neuralnet.ann'},
	{name='ann-ffi', ctor=require 'neuralnet.ann-ffi'},

	-- [[ the C++ version has a slightly dif API since it groups the layer stuff into sub-objects.
	-- I didn't do this in Lua because more tables = more memory and more dereferences
	-- but in C++ this isn't the case
	{name='NeuralNet::ANN<float>', ctor = function(...)
		local nn = require 'NeuralNetLua'['NeuralNet::ANN<float>'](...)		-- float goes maybe 15% faster than double
		nn.input = nn.layers[1].x
		return nn
	end},
	{name='NeuralNet::ANN<double>', ctor = function(...)
		local nn = require 'NeuralNetLua'['NeuralNet::ANN<double>'](...)
		nn.input = nn.layers[1].x		-- make compat with old api
		return nn
	end},
	--]]
} do
	print()
	timer(info.name,function()
		local nn = info.ctor(222, 80, 40, 2)
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
