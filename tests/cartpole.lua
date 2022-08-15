#!/usr/bin/env luajit
--[[
adapted from http://pages.cs.wisc.edu/~finton/rlpage.html
--]]

local class = require 'ext.class'
local math = require 'ext.math'
local gl = require 'gl'
local GLApp = require 'glapp'
local TDNN = require 'neuralnet.tdnn'

math.randomseed(os.time())


--[[
state description
used for generating the neural network
and used for providing input state object 
--]]
local StateDescription = class()

function StateDescription:createNeuralNetwork(...)
	local rlnn = TDNN(...)
	rlnn.gamma = .9
	rlnn.noise = 0
	for k=1,#rlnn.nn.w do
		-- initialize weights to zero
		for i=1,#rlnn.nn.w[k] do
			for j=1,#rlnn.nn.w[k][i] do
				rlnn.nn.w[k][i][j] = 0
			end
		end
	end
	return rlnn
end

local DiscreteStateDescription = class(StateDescription)

function DiscreteStateDescription:createNeuralNetwork(...)
	local rlnn = DiscreteStateDescription.super.createNeuralNetwork(self, ...)
	rlnn.activation = function(x) return x end
	rlnn.activationDeriv = function() return 1 end
	for k=1,#rlnn.nn.w do
		-- disable neural net bias
		rlnn.nn.useBias[k] = false
	end
	return rlnn
end

--[[
	returns 0 for x < min 
	returns n-1 for x > max
	returns even divisions for x from min to max
--]]
function DiscreteStateDescription:bin(x, min, max, n)
	return math.clamp(
		math.floor((x - min) / (max - min) * (n - 2)),
		-1, n-2) + 1
end

DiscreteStateDescription.xBins = 3
DiscreteStateDescription.dxdtBins = 3
DiscreteStateDescription.thetaBins = 6
DiscreteStateDescription.dthetadtBins = 3

DiscreteStateDescription.numStates = 
	DiscreteStateDescription.xBins * 
	DiscreteStateDescription.dxdtBins * 
	DiscreteStateDescription.thetaBins * 
	DiscreteStateDescription.dthetadtBins

DiscreteStateDescription.numActions = 3

function DiscreteStateDescription:getState(x, dx_dt, theta, dtheta_dt)
	if x < -2.4 or x > 2.4 
	or theta < -math.rad(12)
	or theta > math.rad(12)
	then
		return 0	-- invalid state means we've failed
	end

	local xIndex = self:bin(x, -.8, .8, self.xBins)
	local dxdtIndex = self:bin(dx_dt, -.5, .5, self.dxdtBins)

	-- theta is nonlinear...
	assert(self.thetaBins == 6)
	local thetaIndex
	if theta < -math.deg(6) then 
		thetaIndex = 0
	elseif theta < -math.deg(1) then
		thetaIndex = 1
	elseif theta < 0 then
		thetaIndex = 2
	elseif theta < math.deg(1) then
		thetaIndex = 3
	elseif theta < math.deg(6) then
		thetaIndex = 4
	else
		thetaIndex = 5
	end

	local dthetadtIndex = self:bin(dtheta_dt, -math.rad(50), math.rad(50), self.dthetadtBins)

	return 
		xIndex + self.xBins * (
		dxdtIndex + self.dxdtBins * (
		thetaIndex + self.thetaBins * (
		dthetadtIndex)))
		+ 1 	-- +1 for lua
end


local ContinuousStateDescription = class(StateDescription)

ContinuousStateDescription.numActions = 3
ContinuousStateDescription.numStates = 4

function ContinuousStateDescription:createNeuralNetwork(...)
	local desc = self
	
	local rlnn = ContinuousStateDescription.super.createNeuralNetwork(self, ...)
	rlnn.gamma = .99	-- q-learning gamma
	rlnn.alpha = 1		-- weight update amount
	rlnn.lambda = .99	-- history influence
	rlnn.noise = 0		-- noise within the choosing process.  TODO this should be noise of whether to choose the greedy or a random action 
	rlnn.historySize = 100
	
	-- set signal to sigmoid
	--rlnn.activation = function(x) return 1 / (1 + math.exp(-x)) end
	--rlnn.activationDeriv = function(x,y) return y * (1 - y) end
	local oobslope = .3
	rlnn.activation = function(x)
		if x < 0 then return oobslope * x end
		if x > 1 then return oobslope * x + (1 - oobslope) end
		return x
	end
	rlnn.activationDeriv = function(x,y)
		if x < 0 or x > 1 then return oobslope end
		return 1
	end
	
	-- set input to be signals of state + last output (last action)
	rlnn.feedForwardForState = function(self, state)
		assert(#state == desc.numStates)
		assert(#state + desc.numActions == #self.nn.input)
		-- upload states
		for i=1,#state do
			self.nn.input[i] = state[i]
		end
		-- upload last action
		if self.lastAction then
			for i=1,desc.numActions do
				self.nn.input[i+#state] = self.lastAction == i and 1 or 0
			end
		else
			for i=1,desc.numActions do
				self.nn.input[i+#state] = 0 
			end
		end
		self.nn:feedForward()
	end
	return rlnn
end

function ContinuousStateDescription:getState(x, dx_dt, theta, dtheta_dt)
	-- maps input signals to 0,1 range
	local function ramp(x, min, max)
		return math.clamp((x - min) / (max - min), 0, 1)
	end

	-- maps from [-range,range] to [0,1] with bias for center values
	local function powRamp(x, range)
		return math.clamp(math.sign(x) * math.sqrt(math.abs(x / range)) * .5 + .5, 0, 1)
	end
	
	return {
		ramp(x, -2.4, 2.4), 
		ramp(dx_dt, -.5, .5), 
		powRamp(theta, math.rad(6)),	-- bias in the middle:  -6,-1,0,1,6.  use power * sign
		ramp(dtheta_dt, -math.rad(50), math.rad(50))
	}
end


-- init based on whether we want discrete/continuous state representation 
local neuralNetworkMethod = 'singleLayerDiscreteState'
--local neuralNetworkMethod = 'multiLayerDiscreteState'
--local neuralNetworkMethod = 'multiLayerContinuousState'

local getState
local createNeuralNetwork = ({
	-- typical q-learning.  no activation function, no hidden layer
	-- discretize all possible states, the 'state' variable is the nn input is a spike at the current discrete state index
	singleLayerDiscreteState = function()
		
		local desc = DiscreteStateDescription()
		getState = function(...)
			return desc:getState(...)
		end
		
		return desc:createNeuralNetwork(
			desc.numStates,
			desc.numActions)
	end,

	-- keeping the same discrete input system
	--  adding a middle layer just to make sure that the middle layer will still work
	multiLayerDiscreteState = function()
		
		local desc = DiscreteStateDescription()
		getState = function(...)
			return desc:getState(...)
		end
		
		local rlnn = desc:createNeuralNetwork(desc.numStates, desc.numStates, desc.numActions)
		
		-- without initializing it to identity (which makes it redundant) there seems to be a lot of noise ...
		for i=1,#rlnn.nn.w[2] do
			for j=1,#rlnn.nn.w[2][i] do
				rlnn.nn.w[2][i][j] = i == j and 1 or 0
			end
		end
		
		return rlnn
	end,

	-- my attempt at q-learning neural networks
	-- not going so well
	multiLayerContinuousState = function()
		
		local desc = ContinuousStateDescription()
		getState = function(...)
			return desc:getState(...)
		end

		local rlnn = desc:createNeuralNetwork(
			desc.numStates + desc.numActions,
			10,
			10,
			10,
			10,
			10,
			desc.numActions)

		for k=1,#rlnn.nn.w do
			for i=1,#rlnn.nn.w[k] do
				for j=1,#rlnn.nn.w[k][i] do
					rlnn.nn.w[k][i][j] = (math.random() * 2 - 1) * .5	--i == j and 1 or 0
				end
			end
		end

		return rlnn	
	end,	
})[neuralNetworkMethod]

local CartController = class()
function CartController:init()
	self.qnn = createNeuralNetwork()
	self:reset()
end
function CartController:reset()
	self.firstMove = true
	self.qnn.lastAction = nil
end

function CartController:getAction(x, dx_dt, theta, dtheta_dt, reward)
	-- calculate state based on cart parameters
	--  to be used for reinforcement and for determining next action
	local state = getState(x, dx_dt, theta, dtheta_dt)

	--apply reward
	-- don't apply rewards until we have a previous state/action on file
	if not self.firstMove then
		self.qnn:applyReward(state, reward)
	end
	self.firstMove = false

	-- determine next action
	return self.qnn:determineAction(state)
end

local Cart = class()
Cart.cartControllerClass = CartController
Cart.maxIterations = 100000
Cart.maxFailures = 3000
function Cart:init()
	self.controller = self.cartControllerClass()
	self:reset()
end
function Cart:reset()
	self.x = 0
	self.dx_dt = 0
	self.theta = (math.random() * 2 - 1) * math.rad(6)
	self.dtheta_dt = 0
	self.iteration = 0
	self.controller:reset()
end
function Cart:step()
	if self.iteration >= self.maxIterations then
		self:reset()
	else
		self.iteration = self.iteration + 1
		local action = self.controller:getAction(self.x, self.dx_dt, self.theta, self.dtheta_dt, .001)

		local failed = self:simulate(action)

		if failed then
			print(self.iteration)
			io.stdout:flush()
			-- reinforce
			self.controller:getAction(self.x, self.dx_dt, self.theta, self.dtheta_dt, -1)
			self:reset()
		end
	end
end
function Cart:simulate(action)
	local gravity = 9.8
	local massCart = 1
	local massPole = .1
	local totalMass = massPole + massCart
	local length = .5
	local poleMassLength = massPole * length
	local forceMag = 20
	local dt = .02

	local force = ({-forceMag, 0, forceMag})[action]
	local cosTheta = math.cos(self.theta)
	local sinTheta = math.sin(self.theta)
	local temp = (force + poleMassLength * self.dtheta_dt * self.dtheta_dt * sinTheta) / totalMass
	local d2theta_dt2 = (gravity * sinTheta - cosTheta * temp) / (length * (4./3. - massPole * cosTheta * cosTheta / totalMass))
	local d2x_dt2 = temp - poleMassLength * d2theta_dt2 * cosTheta / totalMass

	self.x = self.x + dt * self.dx_dt
	self.theta = self.theta + dt * self.dtheta_dt
	self.dx_dt = self.dx_dt + dt * d2x_dt2
	self.dtheta_dt = self.dtheta_dt + dt * d2theta_dt2

	-- fail?
	if self.theta < -math.rad(12) or self.theta > math.rad(12) then return true end
	if self.x < -2.4 or self.x > 2.4 then return true end
end
local cart = Cart()

local CartPoleGLApp = class(GLApp)
function CartPoleGLApp:update()
	gl.glClear(gl.GL_COLOR_BUFFER_BIT)

	cart:step()

	local width, height = self:size()
	local aspectRatio = width / height

	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	local scale = 3
	gl.glOrtho(-aspectRatio * scale, aspectRatio * scale, -scale, scale, -1, 1)

	gl.glMatrixMode(gl.GL_MODELVIEW)
	gl.glLoadIdentity()

	gl.glPointSize(3)
	gl.glColor3f(1,0,0)
	gl.glBegin(gl.GL_POINTS)
	gl.glVertex2f(cart.x, 0)
	gl.glEnd()
	gl.glColor3f(1,1,0)
	gl.glBegin(gl.GL_LINES)
	gl.glVertex2f(cart.x, 0)
	gl.glVertex2f(cart.x + math.sin(cart.theta), math.cos(cart.theta))
	gl.glEnd()

	local function colorForQ(q)
		return .5 + math.max(0,q), .5, .5 - math.min(0,q)
	end
	gl.glPushMatrix()
	gl.glTranslatef(-4, -2.5, 0)
	gl.glScalef(.2, .2, .1)
	gl.glBegin(gl.GL_QUADS)
	if #cart.controller.qnn.nn.w[1][1] == 3*3*6*3+1 then
		for i=0,3-1 do
			for j=0,3-1 do
				for k=0,6-1 do
					for l=0,3-1 do
						for action=1,3 do
							local state = 1 + i + 3 * (j + 3 * (k + 6 * l))
							gl.glColor3f(colorForQ(cart.controller.qnn.nn.w[1][action][state]))
							gl.glVertex2f(7*i + k + .2 * (action-1), 4*j + l)
							gl.glVertex2f(7*i + k + .2 * (action-1 + .8), 4*j + l)
							gl.glVertex2f(7*i + k + .2 * (action-1 + .8), 4*j + l+.9)
							gl.glVertex2f(7*i + k + .2 * (action-1), 4*j + l+.9)
						end
					end
				end
			end
		end
	end
	gl.glEnd()
	gl.glPopMatrix()
end
CartPoleGLApp():run()
