#!/usr/bin/env luajit
--[[
this is feeding in a determined set of numbers and used for verifying that the output is all matching for each of my networks
--]]
local table = require 'ext.table'
local ANN = require 'neuralnet.ann'
local ANNFFI = require 'neuralnet.ann-ffi'
local ANNCPP = require 'neuralnet.ann-cpp'

local function makeCpp(name)
	return {name=name, ctor=ANNCPP(name)}
end

local cols = table{
	--{['input'] = function(nn) return nn.input end},
	{['input L1 norm'] = function(nn) return nn.input:normL1() end},
	--{['w[1] '] = function(nn) return nn.w[1])
	{['w[1] L1 norm'] = function(nn) return nn.w[1]:normL1() end},
	--{['x[2] '] = function(nn) return nn.x[2])
	{['x[2] L1 norm'] = function(nn) return nn.x[2]:normL1() end},
	--{['w[2] '] = function(nn) return nn.w[2])
	{['w[2] L1 norm'] = function(nn) return nn.w[2]:normL1() end},
	--{['x[3] '] = function(nn) return nn.x[3])
	{['x[3] L1 norm'] = function(nn) return nn.x[3]:normL1() end},
	--{['w[3] '] = function(nn) return nn.w[3])
	{['w[3] L1 norm'] = function(nn) return nn.w[3]:normL1() end},
	--{['output'] = function(nn) return nn.output)
	{['output L1 norm'] = function(nn) return nn.output:normL1() end},

	--{['desired'] = function(nn) return nn.desired)
	{['desired L1 norm'] = function(nn) return nn.desired:normL1() end},
	{['total error'] = function(nn, totalError) return totalError end},
	--{['outputError'] = function(nn) return nn.outputError)
	{['outputError L1 norm'] = function(nn) return nn.outputError:normL1() end},
	--{['netErr[3]'] = function(nn) return nn.netErr[3])
	{['netErr[3] L1 norm'] = function(nn) return nn.netErr[3]:normL1() end},
	--{['xErr[3]'] = function(nn) return nn.xErr[3])
	{['xErr[3] L1 norm'] = function(nn) return nn.xErr[3]:normL1() end},
	--{['w[3] '] = function(nn) return nn.w[3])
	{['w[3] L1 norm'] = function(nn) return nn.w[3]:normL1() end},
	--{['netErr[2]'] = function(nn) return nn.netErr[2])
	{['netErr[2] L1 norm'] = function(nn) return nn.netErr[2]:normL1() end},
	--{['xErr[2]'] = function(nn) return nn.xErr[2])
	{['xErr[2] L1 norm'] = function(nn) return nn.xErr[2]:normL1() end},
	--{['w[2] '] = function(nn) return nn.w[2])
	{['w[2] L1 norm'] = function(nn) return nn.w[2]:normL1() end},
	--{['netErr[1]'] = function(nn) return nn.netErr[1])
	{['netErr[1] L1 norm'] = function(nn) return nn.netErr[1]:normL1() end},
	--{['xErr[1]'] = function(nn) return nn.xErr[1])
	{['xErr[1] L1 norm'] = function(nn) return nn.xErr[1]:normL1() end},
	--{['w[1] '] = function(nn) return nn.w[1])
	{['w[1] L1 norm'] = function(nn) return nn.w[1]:normL1() end},
}
local reports = table()

for _,info in ipairs{
	{name='neuralnet.ann', ctor=ANN},
	{name='neuralnet.ann-ffi', ctor=ANNFFI},
	makeCpp'NeuralNet::ANN<float>',
	makeCpp'NeuralNet::ANN<double>',
	makeCpp'NeuralNet::ANN<std::float32_t>',
	makeCpp'NeuralNet::ANN<std::float64_t>',
	makeCpp'NeuralNet::ANN<std::float16_t>',
	makeCpp'NeuralNet::ANN<std::bfloat16_t>',
	--makeCpp'NeuralNet::ANN<long double>',	-- segfaults ...
	--makeCpp'NeuralNet::ANN<std::float128_t>',	-- segfaults ...
} do
	local i = 0
	local function src()
		i = i + 1
		return math.sin(1/i)
		--return math.random() * 2 - 1
	end

	--local nn = info.ctor(222, 80, 40, 2)
	local nn = info.ctor(5,4,3,2)
	nn:setActivation'identity'
	nn:setActivationDeriv'one'
	for k,w in ipairs(nn.w) do
		--local height, width = w:size():unpack()
		-- for cpp compat 
		local height = #w
		local width = #w[1]
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
	
	nn.desired[1] = src()
	nn.desired[2] = src()
	local totalError = nn:calcError()
	nn:backPropagate()

	local report = table()
	report.name = info.name
	for i,col in ipairs(cols) do
		local k,v = next(col)
		report[i] = v(nn, totalError)
	end
	reports:insert(report)
end

for _,report in ipairs(reports) do
	print()
	print(report.name)
	for i,col in ipairs(cols) do
		local k,v = next(col)
		print(k,
			('%.5f%% error'):format(
				math.abs(report[i] - reports[1][i]) / reports[1][i] * 100
			),
			 report[i]
		)
	end
end
