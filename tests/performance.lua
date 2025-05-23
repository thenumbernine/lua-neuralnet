#!/usr/bin/env luajit
--[[
performance test between feedForward and backprop, with and without batch, matrix vs matrix.ffi vs maybe I'll write my own wrappers to Blaze or something

.. the more i try to optimize ann-ffi the slower it goes
so ann-ffi is only beneficial for larger networks
and any attempt to 'optimize' it makes it go slower
luajit is weird

the naive c++ impl using row-major i.e. dot-products per-col-elem ran as fast as luajit
but expanding 4x of the inner loop made gcc implciitly pick up simd and i got a ... 3x ... perf increase
from there, i exposed all the c++23 float types ... and all my float16's go dog slow.
float and double are the fastest.  float is about 10% faster.
maybe I will get into the avx functions or something idk
--]]
local timer = require 'ext.timer'.timerQuiet
local numIter = 10000

local function test(info)
	local ffTime
	local ffbpTime
	local totalTime = timer(function()
		--local nn = info.ctor(222, 80, 40, 2)
		local nn = info.ctor(104, 40, 40, 40, 2)
		--nn.useBatch = 25
		
		-- TODO how much is filling the input with __index vs ptr access changing things?
		-- welp filling is 1/100th the time of feedForward and backProp so :shrug:
		--for i=1,#nn.input do	-- ann-cpp...
		for i=1,#nn.w[1][1]-1 do
			nn.input[i] = math.random()
		end
		ffTime = timer(function()
			for i=1,numIter do
				nn:feedForward()
			end
		end)
		ffbpTime = timer(function()
			for i=1,numIter do
				nn:feedForward()
				nn.desired[1] = math.random()
				nn.desired[2] = math.random()
				nn:calcError()
				nn:backPropagate()
			end
		end)
	end)
	print(('%q'):format(info.name), ffTime, ffbpTime, totalTime)
end

test{name='ann', ctor=require 'neuralnet.ann'}

-- hmm ann-ffi isn't much faster as float
-- maybe even 5% slower for some reason
require 'matrix.ffi'.real = 'float'
test{name='ann-ffi[float]', ctor=require 'neuralnet.ann-ffi'}

require 'matrix.ffi'.real = 'double'
test{name='ann-ffi[double]', ctor=require 'neuralnet.ann-ffi'}

-- [[ the C++ version has a slightly dif API since it groups the layer stuff into sub-objects.
-- I didn't do this in Lua because more tables = more memory and more dereferences
-- but in C++ this isn't the case
for _,name in ipairs{
	'NeuralNet::ANN<float>',
	'NeuralNet::ANN<double>',
	--[[ these now only work with nospeedhacks set
	'NeuralNet::ANN<std::float32_t>',
	'NeuralNet::ANN<std::float64_t>',
	'NeuralNet::ANN<std::float16_t>',
	'NeuralNet::ANN<std::bfloat16_t>',
	--]]
	--'NeuralNet::ANN<long double>',	-- segfaults ...
	--'NeuralNet::ANN<std::float128_t>',
} do
	test{
		name = name,
		ctor = function(...)
			return require 'neuralnet.ann-cpp'(name, 'nospeedhacks')(...)
		end,
	}
end
--]]
