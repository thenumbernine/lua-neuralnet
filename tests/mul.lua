#!/usr/bin/env luajit
--[[
I want a problem that I can watch convergence upon (like and or xor etc)
but maybe a bit bigger so i can compare integration methods
how about multiply in binary form?
--]]

local table = require 'ext.table'
local range = require 'ext.range'
local gnuplot = require 'gnuplot'
--local ANN = require 'neuralnet.ann'
local ANN = require 'neuralnet.ann-ffi'

--[=[ C++ / compat layer ... TODO I do this often enough
local function ANN(...)
	local NNLib = require 'NeuralNetLua'
	local nn = NNLib['NeuralNet::ANN<float>'](...)
	nn.input = nn.layers[1].x
	-- until I do this for the lua versions ... here's a compat layer:
	nn.x = {}
	nn.xErr = {}
	nn.net = {}
	nn.netErr = {}
	nn.w = {}
	nn.dw = {}		
	for i,layer in ipairs(nn.layers) do
		nn.x[i] = assert(layer.x)
		nn.xErr[i] = assert(layer.xErr)
		nn.net[i] = assert(layer.net)
		nn.netErr[i] = layer.netErr
		nn.w[i] = assert(layer.w)
		nn.dw[i] = assert(layer.dw)
		-- only for tostring.  TODO implement that in C++
		local w = nn.w[i]
		function w:toLuaMatrix()
			return require 'matrix'.lambda({w:height(), w:width()}, function(i,j) return w[i][j] end)
		end
	end
	nn.x[#nn.layers+1] = nn.output
	nn.xErr[#nn.layers+1] = nn.outputError
	return nn
end
--]=]

--math.randomseed(os.time())
math.randomseed(1)

local n = 3

local nn = ANN(2*n, 2*n, n)
nn.useBatch = bit.lshift(1,n)
nn.dt = .1 / nn.useBatch

local hi, lo = -.9, .9

local results = table()

for iter=1,1000000 do
	local a = math.random(0, 0xff)
	local b = math.random(0, 0xff)
	local c = bit.band(a * b, 0xff)
	for i=0,n-1 do
		nn.input[i+1] = bit.band(1, bit.rshift(a, i)) > .5 and hi or lo
		nn.input[i+1+n] = bit.band(1, bit.rshift(b, i)) > .5 and hi or lo
		nn.desired[i+1] = bit.band(1, bit.rshift(c, i)) > .5 and hi or lo
	end
	nn:feedForward()
	local err = nn:calcError()
	results:insert(err)
	-- [[ typical backprop / newton descent
	nn:backPropagate()
	--]]
	--[[ TODO krylov subspace solver, A(x) = backprop on the weights (proly without batch ... ?)
	-- we're solving x for A x = b
	-- x = weights at next state
	-- A = backprop
	-- b = weigths at prev step
	require 'solver.conjres'{
		A = function()
		end,
	}()
	-- lets just try a pred-corr solver first, they do better for nonlinear stuff anyways
	-- wait, is pred-corr just the same as performing grdient descent for a fixed input/desired until it fully converges?
	-- yeahhhh... pred-corr is just converging to the infinite limit of a single state configuration ... maybe it' be more successful with batch? 
	-- but I think pred-corr isn't good for backprop
	--]]
end

gnuplot{
	terminal = 'png size 1024,768',
	output = 'mul.png',
	style = 'data lines',
	log = 'y',
	data = {results},
	{using='0:1', title='error'},
}

print'layer 1 weights'
print(nn.w[1]:toLuaMatrix())
print'layer 2 weights'
print(nn.w[2]:toLuaMatrix())
