#!/usr/bin/env luajit
local ANN = require 'neuralnet.ann'
local ANNFFI = require 'neuralnet.ann-ffi'
local numIter = 10000
for _,info in ipairs{
	{name='ann', cl=ANN},
	{name='ann-ffi', cl=ANNFFI},
} do
	local i = 0
	local function src()
		i = i + 1
		return math.sin(1/i)
	end

	local nn = info.cl(222, 80, 40, 2)
	for k,w in ipairs(nn.w) do
		local height, width = w:size():unpack()
		for i=1,height do
			for j=1,width do
				w[i][j] = src()
			end
		end
	end
	-- TODO how much is filling the input with __index vs ptr access changing things?
	-- welp filling is 1/100th the time of feedForward and backProp so :shrug:
	for i=1,#nn.input do
		nn.input[i] = src()
	end
	for i=1,numIter do
		nn:feedForward()
		nn.desired[1] = src()
		nn.desired[2] = src()
		nn:backPropagate()
	end
	for i=1,numIter do
		nn:feedForward()
	end
	
	print()
	print('input', nn.input)
	print('input L1 norm', nn.input:normL1())
	print('hidden', nn.x[2])
	print('hidden L1 norm', nn.x[2]:normL1())
	print('hidden', nn.x[3])
	print('hidden L1 norm', nn.x[3]:normL1())
	print('output', nn.output)
	print('output L1 norm', nn.output:normL1())
end
