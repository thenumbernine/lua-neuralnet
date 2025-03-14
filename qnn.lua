local ANN = require 'neuralnet.ann'
local class = require 'ext.class'
local table = require 'ext.table'

-- q-learner with NN impl
local QNN = class()

QNN.gamma = .9
QNN.alpha = .5
QNN.noise = .1

function QNN:init(...)
	local layerSizes = {...}
	self.nn = ANN(...)
	-- TODO disable all biases?
	-- TODO clear initial weights?
	-- TODO use linear activation function?
end

--[[
This function is responsible for translating the state into neuron input signals.
This is the one function that has to be replaced if you are going to change state from an integer to anything else.
--]]
function QNN:feedForwardForState(state)
	-- TODO high/low signal values?
	for i=1,#self.nn.input do
		self.nn.input[i] = 0
	end
	if 1 <= state and state <= #self.nn.input then
		self.nn.input[state] = 1
	end
	self.nn:feedForward()
end

-- function QNN:getReward(state, action, nextState)
-- function QNN:applyAction(state, action)
function QNN:getBestAction(state, noise)
	self:feedForwardForState(state)
	local qs = self.nn.output
	
	noise = noise or self.noise or 0
	local bestAction = 1
	local bestValue = qs[1] + math.random() * noise
	for i=2,#qs do
		local checkValue = qs[i] + math.random() * noise
		if bestValue < checkValue then
			bestValue = checkValue
			bestAction = i
		end
	end
	return bestAction, qs[bestAction]
end

function QNN:determineAction(state, noise)
	-- S[t] is our current state, represented as 'state'

	-- A[t] is our action for this state.  get it by getting the Q's of our current state, permuting them sightly, and picking the highest action
	local action, actionQ = self:getBestAction(state, noise)		-- A[t], Q(S[t], A[t])

	self.lastState = state
	self.lastAction = action
	self.lastStateActionQ = actionQ

	return action, actionQ
end

function QNN:applyReward(newState, reward, lastState, lastAction, lastStateActionQ)
	local maxNextQ = select(2, self:getBestAction(newState, 0))			-- max(Q(S[t+1], *))

	-- setup input for backpropagation
	self:feedForwardForState(lastState)
	for i=1,#self.nn.outputError do
		self.nn.outputError[i] = 0
	end
	local err = reward + self.gamma * maxNextQ - lastStateActionQ
	self.nn.outputError[lastAction] = err
	self.nn:backPropagate(self.alpha)

	return err
end

--[[
controller provides:
	:reset()
	:getState()
	:performAction()
	:getReward()
--]]
function QNN:step(controller)
	-- calculate state based on cart parameters
	--  to be used for reinforcement and for determining next action
	-- ok 'self' really is the state
	-- while 'state' is the underlying neural network's representation of the state
	local state = controller:getState()

	-- determine next action.
	-- this also saves it as 'lastAction' and its Q-value as 'lastStateActionQ' for the next 'applyReward'
	-- should this go here or only after the failed condition?
	-- here means no need to store and test for lastAction anymore...
	local action, actionQ = self:determineAction(state)

	controller:performAction(action, actionQ, state)
	local newState = controller:getState()

	-- determine reward and whether to reset
	local reward, reset = controller:getReward()

	--apply reward
	-- applies reward with qnn.lastAction as the A(S[t],*) and lastActionQ
	self:applyReward(newState, reward, state, action, actionQ)

	if reset then
		controller:reset()
	end

	return reward, reset
end

function QNN:getQs(state)
	self:feedForwardForState(state)
	return setmetatable({unpack(self.nn.output)}, table)
end

function QNN:__tostring()
	local qsForSs = table()
	for i=1,#self.nn.input do
		local qs = self:getQs(i)
		qsForSs:insert(require 'ext.tolua'(qs))
	end
	return qsForSs:concat(' ')
end

function QNN.__concat(a,b) return tostring(a) .. tostring(b) end

return QNN
