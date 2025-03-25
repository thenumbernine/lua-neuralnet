#!/usr/bin/env luajit
--[[
adapted from http://pages.cs.wisc.edu/~finton/rlpage.html
--]]

local class = require 'ext.class'
local math = require 'ext.math'
local TDEnv = require 'neuralnet.tdnn'

math.randomseed(os.time())


local State = class()
function State:init()
	self.x = 0
	self.dt_x = 0
	self.theta = (math.random() * 2 - 1) * math.rad(6)
	self.dt_theta = 0
	self.iteration = 0
end


--[[
state description
used for generating the neural network
and used for providing input state object
--]]
local StateDescription = class()

function StateDescription:createNeuralNetwork(...)
	local rlnn = TDEnv(...)
	rlnn.alpha = .1
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

local DiscreteStateDescription = StateDescription:subclass()

function DiscreteStateDescription:createNeuralNetwork(...)
	local rlnn = DiscreteStateDescription.super.createNeuralNetwork(self, ...)
	--rlnn.historySize = 100
	rlnn.noise = 0 --1e-5	-- tempting to set this for initial gains, but in the long run it always hinders things
	rlnn.nn:setActivation'identity'
	rlnn.nn:setActivationDeriv'one'
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
function bin(x, min, max, n)
	return math.clamp(
		math.floor((x - min) / (max - min) * (n - 2)),
		-1, n-2) + 1
end

DiscreteStateDescription.xBins = 3
DiscreteStateDescription.dtxBins = 3
DiscreteStateDescription.thetaBins = 6
DiscreteStateDescription.dtthetaBins = 3

DiscreteStateDescription.numStates =
	DiscreteStateDescription.xBins *
	DiscreteStateDescription.dtxBins *
	DiscreteStateDescription.thetaBins *
	DiscreteStateDescription.dtthetaBins

DiscreteStateDescription.numActions = 3

function DiscreteStateDescription:getStateIndex(x, dt_x, theta, dt_theta)
	if x < -2.4 or x > 2.4
	or theta < -math.rad(12)
	or theta > math.rad(12)
	then
		return 0	-- invalid state means we've failed
	end

	local xIndex = bin(x, -.8, .8, self.xBins)
	local dxdtIndex = bin(dt_x, -.5, .5, self.dtxBins)

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

	local dthetadtIndex = bin(dt_theta, -math.rad(50), math.rad(50), self.dtthetaBins)

	return
		xIndex + self.xBins * (
		dxdtIndex + self.dtxBins * (
		thetaIndex + self.thetaBins * (
		dthetadtIndex)))
		+ 1 	-- +1 for lua
end


local ContinuousStateDescription = StateDescription:subclass()

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
	-- "leaky" "hard" sigmoid ... linear approx of sigmoid w/extra slopes
	local oobslope = .3
	rlnn.nn:setActivation(function(x)
		if x < 0 then return oobslope * x end
		if x > 1 then return oobslope * x + (1 - oobslope) end
		return x
	end)
	rlnn.nn:setActivationDeriv(function(x,y)
		if x < 0 or x > 1 then return oobslope end
		return 1
	end)

	-- set input to be signals of state + last output (last action)
	rlnn.feedForwardForState = function(self, state)
		assert(#state == desc.numStates)
		assert(#state + desc.numActions == #self.nn.input)
		-- upload states
		for i=1,#state do
			self.nn.input[i] = state[i]
		end
error "FIXME self.lastAction isn't stored anymore"
		if self.lastAction then
			for i=1,desc.numActions do
				self.nn.input[i+#state] = self.lastAction == i and 1 or 0
			end
		else
			for i=1,desc.numActions do
				self.nn.input[i+#state] = 0
			end
		end
		--]]
		self.nn:feedForward()
	end
	return rlnn
end

function ContinuousStateDescription:getStateIndex(x, dt_x, theta, dt_theta)
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
		ramp(dt_x, -.5, .5),
		powRamp(theta, math.rad(6)),	-- bias in the middle:  -6,-1,0,1,6.  use power * sign
		ramp(dt_theta, -math.rad(50), math.rad(50))
	}
end


-- init based on whether we want discrete/continuous state representation
local neuralNetworkMethod = 'singleLayerDiscreteState'
--local neuralNetworkMethod = 'multiLayerDiscreteState'
--local neuralNetworkMethod = 'multiLayerContinuousState'

local getStateIndex
local createNeuralNetwork = ({
	-- typical q-learning.  no activation function, no hidden layer
	-- discretize all possible states, the 'state' variable is the nn input is a spike at the current discrete state index
	singleLayerDiscreteState = function()

		local desc = DiscreteStateDescription()
		getStateIndex = function(agent, state)
			return desc:getStateIndex(state.x, state.dt_x, state.theta, state.dt_theta)
		end

		return desc:createNeuralNetwork(
			desc.numStates,
			desc.numActions)
	end,

	-- keeping the same discrete input system
	--  adding a middle layer just to make sure that the middle layer will still work
	multiLayerDiscreteState = function()

		local desc = DiscreteStateDescription()
		getStateIndex = function(agent, state)
			return desc:getStateIndex(state.x, state.dt_x, state.theta, state.dt_theta)
		end

		local rlnn = desc:createNeuralNetwork(desc.numStates, desc.numStates, desc.numActions)

		-- [[ without initializing it to identity (which makes it redundant) there seems to be a lot of noise ...
		for i=1,#rlnn.nn.w[2] do
			for j=1,#rlnn.nn.w[2][i] do
				--rlnn.nn.w[2][i][j] = i == j and 1 or 0
				rlnn.nn.w[2][i][j] = (math.random() - .5) * .1
			end
		end
		--]]

		return rlnn
	end,

	-- my attempt at q-learning neural networks
	-- not going so well
	multiLayerContinuousState = function()

		local desc = ContinuousStateDescription()
		getStateIndex = function(agent, state)
			return desc:getStateIndex(state.x, state.dt_x, state.theta, state.dt_theta)
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

local Agent = class()
Agent.successIterations = 100000
Agent.maxFailures = 3000
function Agent:init()
	self.qnn = createNeuralNetwork()
	Agent.getStateIndex = getStateIndex
	self:resetState()
end
function Agent:resetState()
	if self.state then print(self.state.iteration) end
	self.state = State()
	return self.state
end
function Agent:observe(state)
	local stateIndex = self:getStateIndex(state)

	local nn = self.qnn.nn
	for i=1,#nn.input do
		nn.input[i] = 0
	end
	if 1 <= stateIndex and stateIndex <= #nn.input then
		nn.input[stateIndex] = 1
	end
end
function Agent:performAction(state, action, actionQ)	-- 'action' is really all we want
	-- step / performAction
	local gravity = 9.8
	local massCart = 1
	local massPole = .1
	local totalMass = massPole + massCart
	local length = .5
	local poleMassLength = massPole * length
	local forceMag = 20
	local dt = .02

	local force = 0
	if action == 2 then
		force = -forceMag
	elseif action == 3 then
		force = forceMag
	end

	local cosTheta = math.cos(state.theta)
	local sinTheta = math.sin(state.theta)
	local temp = (force + poleMassLength * state.dt_theta * state.dt_theta * sinTheta) / totalMass
	local d2t_theta = (gravity * sinTheta - cosTheta * temp) / (length * (4./3. - massPole * cosTheta * cosTheta / totalMass))
	local d2t_x = temp - poleMassLength * d2t_theta * cosTheta / totalMass

	local newState = State()
	newState.x = state.x + dt * state.dt_x
	newState.theta = state.theta + dt * state.dt_theta
	newState.dt_x = state.dt_x + dt * d2t_x
	newState.dt_theta = state.dt_theta + dt * d2t_theta
	newState.iteration = state.iteration + 1
	return newState
end
function Agent:getReward(state)
	local fail = state.theta < -math.rad(12) or state.theta > math.rad(12) or state.x < -2.4 or state.x > 2.4
	local succeeded = state.iteration >= self.successIterations
	return fail and -1 or .001, fail or succeeded
end
function Agent:step()
	self.state = self.qnn:step(self, self.state)
end
local agent = Agent()

-- [=[ gl display
local gl = require 'gl'
local CartPoleGLApp = require 'glapp.view'.apply(require 'glapp')
function CartPoleGLApp:initGL()
	solidSceneObj = require 'gl.sceneobject'{
		program = {
			version = 'latest',
			precision = 'best',
			vertexCode = [[
in vec3 vertex;
uniform mat4 mvProjMat;
void main() {
	gl_Position = mvProjMat * vec4(vertex, 1.);
}
]],
			fragmentCode = [[
out vec4 fragColor;
uniform vec4 color;
void main() {
	fragColor = color;
}
]],
		},
		geometry = {
			mode = gl.GL_TRIANGLES,	-- will swap this
		},
		vertexes = {
			type = gl.GL_FLOAT,
			dim = 3,
			useVec = true,
		},
		uniforms = {
			color = {1,1,1,1},
			mvProjMat = self.view.mvProjMat.ptr,
		},
	}

	self.view.ortho = true
	self.view.orthoSize = 3
	self.view.znear = -1
	self.view.zfar = 1
	self.view.pos:set(0,0,0)
	self.view.angle:set(0,0,0,1)
end
function CartPoleGLApp:update()
	local view = self.view
	view:setup(self.width / self.height)
	gl.glClear(gl.GL_COLOR_BUFFER_BIT)

	agent:step()

	local width, height = self:size()
	local aspectRatio = width / height

	gl.glPointSize(3)
	solidSceneObj.geometry.mode = gl.GL_POINTS
	solidSceneObj.uniforms.color = {1,0,0,1}
	local vtxs = solidSceneObj:beginUpdate()
	vtxs:emplace_back():set(agent.state.x, 0, 0)
	solidSceneObj:endUpdate()

	solidSceneObj.geometry.mode = gl.GL_LINES
	solidSceneObj.uniforms.color = {1,1,0,1}
	local vtxs = solidSceneObj:beginUpdate()
	vtxs:emplace_back():set(agent.state.x, 0, 0)
	vtxs:emplace_back():set(agent.state.x + math.sin(agent.state.theta), math.cos(agent.state.theta), 0)
	solidSceneObj:endUpdate()

	local function colorForQ(q)
		return .5 + math.max(0,q), .5, .5 - math.min(0,q), 1
	end
	view.mvMat:applyTranslate(-2, -2.5, 0)
	view.mvMat:applyScale(.2, .2, .1)
	view.mvProjMat:mul4x4(view.projMat, view.mvMat)
	solidSceneObj.geometry.mode = gl.GL_TRIANGLES
	if #agent.qnn.nn.w[1][1] == 3*3*6*3+1 then
		for i=0,3-1 do
			for j=0,3-1 do
				for k=0,6-1 do
					for l=0,3-1 do
						for action=1,3 do
							local state = 1 + i + 3 * (j + 3 * (k + 6 * l))
							solidSceneObj.uniforms.color = {colorForQ(agent.qnn.nn.w[1][action][state])}
							local vtxs = solidSceneObj:beginUpdate()
							vtxs:emplace_back():set(7*i + k + .2 * (action-1), 4*j + l, 0)
							vtxs:emplace_back():set(7*i + k + .2 * (action-1 + .8), 4*j + l, 0)
							vtxs:emplace_back():set(7*i + k + .2 * (action-1 + .8), 4*j + l+.9, 0)

							vtxs:emplace_back():set(7*i + k + .2 * (action-1 + .8), 4*j + l+.9, 0)
							vtxs:emplace_back():set(7*i + k + .2 * (action-1), 4*j + l+.9, 0)
							vtxs:emplace_back():set(7*i + k + .2 * (action-1), 4*j + l, 0)
							solidSceneObj:endUpdate()
						end
					end
				end
			end
		end
	end
end
return CartPoleGLApp():run()
--]=]
--[=[ cli
while true do agent:step() end
--]=]
