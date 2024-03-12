#!/usr/bin/env luajit
local ANN = require 'neuralnet.ann'
local ANNFFI = require 'neuralnet.ann-ffi'
for _,info in ipairs{
	{name='ann', cl=ANN},
	{name='ann-ffi', cl=ANNFFI},
} do
	local i = 0
	local function src()
		i = i + 1
		return math.sin(1/i)
		--return math.random() * 2 - 1
	end

	--local nn = info.cl(222, 80, 40, 2)
	local nn = info.cl(5,4,3,2)
	nn.activation = function(x) return x end
	nn.activationDeriv = function(x) return 1 end
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
	
	nn:feedForward()
	
	print()
	--print('input', nn.input)
	print('input L1 norm', nn.input:normL1())
	--print('w[1] ', nn.w[1])
	print('w[1] L1 norm', nn.w[1]:normL1())
	--print('x[2] ', nn.x[2])
	print('x[2] L1 norm', nn.x[2]:normL1())
	--print('w[2] ', nn.w[2])
	print('w[2] L1 norm', nn.w[2]:normL1())
	--print('x[3] ', nn.x[3])
	print('x[3] L1 norm', nn.x[3]:normL1())
	--print('w[3] ', nn.w[3])
	print('w[3] L1 norm', nn.w[3]:normL1())
	--print('output', nn.output)
	print('output L1 norm', nn.output:normL1())

	nn.desired[1] = src()
	nn.desired[2] = src()
	--print('desired', nn.desired)
	print('desired L1 norm', nn.desired:normL1())
	local totalError = nn:calcError()
	print('total error', totalError)
	--print('outputError', nn.outputError)
	print('outputError L1 norm', nn.outputError:normL1())
	nn:backPropagate()
	--print('netErr[3]', nn.netErr[3])
	print('netErr[3] L1 norm', nn.netErr[3]:normL1())
	--print('xErr[3]', nn.xErr[3])
	print('xErr[3] L1 norm', nn.xErr[3]:normL1())
	--print('w[3] ', nn.w[3])
	print('w[3] L1 norm', nn.w[3]:normL1())
	--print('netErr[2]', nn.netErr[2])
	print('netErr[2] L1 norm', nn.netErr[2]:normL1())
	--print('xErr[2]', nn.xErr[2])
	print('xErr[2] L1 norm', nn.xErr[2]:normL1())
	--print('w[2] ', nn.w[2])
	print('w[2] L1 norm', nn.w[2]:normL1())
	--print('netErr[1]', nn.netErr[1])
	print('netErr[1] L1 norm', nn.netErr[1]:normL1())
	--print('xErr[1]', nn.xErr[1])
	print('xErr[1] L1 norm', nn.xErr[1]:normL1())
	--print('w[1] ', nn.w[1])
	print('w[1] L1 norm', nn.w[1]:normL1())
end
