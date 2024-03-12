#!/usr/bin/env luajit
--[[
performance test between feedForward and backprop, with and without batch, matrix vs matrix.ffi vs maybe I'll write my own wrappers to Blaze or something

Press ENTER or type command to continue
ann...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (2.5935509204865s)
 feedForward only...
 ...done feedForward only (0.52986693382263s)
...done ann (3.1246049404144s)

ann-ffi...
 feedForward + backPropagate...
 ...done feedForward + backPropagate (1.267128944397s)
 feedForward only...
 ...done feedForward only (0.38383889198303s)
...done ann (1.6558198928833s)

so ann-ffi is only beneficial for larger networks
--]]
local timer = require 'ext.timer'.timer
local ANN = require 'neuralnet.ann'
local ANNFFI = require 'neuralnet.ann-ffi'
local numIter = 10000
timer('ann',function()
	local nn = ANNFFI(222, 80, 40, 2)
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
			nn:backPropagate()
		end
	end)
	timer('feedForward only', function()
		for i=1,numIter do
			nn:feedForward()
		end
	end)
end)
