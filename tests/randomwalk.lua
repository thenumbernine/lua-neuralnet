#!/usr/bin/env luajit

-- random walk RL
local maxiter = ... or 5

math.randomseed(os.time())

local size = 11
local qnn = require 'neuralnet.tdnn'(size, 2)	qnn.historySize = 100
--local qnn = require 'neuralnet.qnn'(size, 2)
qnn.alpha = .1
qnn.gamma = .99
qnn.noise = 1e-5
qnn.nn.useBias[1] = false
-- [[	
qnn.nn.activation = function(x) return x end
qnn.nn.activationDeriv = function() return 1 end
for i=1,size do
	qnn.nn.w[1][1][i] = 0
	qnn.nn.w[1][2][i] = 0
end
--]]	
local function printState(state)
	for i=1,size do
		local s = ('%.2f/%2.f'):format(qnn.nn.w[1][1][i], qnn.nn.w[1][2][i])
		if i == state then s = '['..s..']' else s = ' '..s..' ' end
		io.write(' | '..s)
	end
	print()
end

local class = require 'ext.class'
local RandomWalk = class()
function RandomWalk:init()
	self:reset()
end
function RandomWalk:reset()
	self.state = math.ceil(size/2)
end
function RandomWalk:getState()	-- held by QNN or held by Controller?
	return self.state
end
function RandomWalk:performAction(action)
	if action == 1 then 
		self.state = self.state-1 
	elseif action == 2 then 
		self.state = self.state+1 
	else
		error('got bad action: '..tostring(action))
	end
end
function RandomWalk:getReward()
	local reward = 0
	local reset
	if self.state == 0 then
		reward = -1
		reset = true
	elseif self.state == size+1 then
		reward = 1
		reset = true
	end
	return reward, reset
end

local controller = RandomWalk()

for iter=1,maxiter do
	controller:reset() 
	local numSteps = 0
	local reward, reset
	repeat
printState(controller.state)
		reward, reset = qnn:step(controller)
		numSteps = numSteps + 1
	until reset
printState(controller.state)
	print(qnn..' steps:'..numSteps)
	print()
end
