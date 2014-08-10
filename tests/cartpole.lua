#!/usr/bin/env luajit

--[[
adapted from http://pages.cs.wisc.edu/~finton/rlpage.html
--]]

local gl = require 'ffi.OpenGL'
local openglapp = require 'openglapp'
local QNN = require 'neuralnet.qnn'

local function printf(...) return print(string.format(...)) end

math.randomseed(os.time())

local xBins = 3
local dxdtBins = 3
local thetaBins = 6
local dthetadtBins = 3
local numStates = xBins * dxdtBins * thetaBins * dthetadtBins
local numActions = 2

local CartController = class()
function CartController:init()
	self.qnn = QNN(numStates, numActions)
	self.qnn.activation = function(x) return x end
	self.qnn.activationDeriv = function() return 1 end
	self.qnn.gamma = .99
	self.qnn.noise = 0
	-- disable neural net bias
	self.qnn.nn.useBias[1] = false
	-- initialize weights to zero
	for i=1,numStates do
		for j=1,numActions do
			self.qnn.nn.w[1][j][i] = 0
		end
	end
	self:reset()
end
function CartController:reset()
	self.curState = 1
	self.prevState = 1
	self.curAction = 0
	self.prevAction = 0
end

--[[
	returns 0 for x < min 
	returns n-1 for x > max
	returns even divisions for x from min to max
--]]
local function bin(x, min, max, n)
	return math.clamp(
		math.floor((x - min) / (max - min) * (n - 2)),
		-1, n-2) + 1
end

local function getState(x, dx_dt, theta, dtheta_dt)
	if x < -2.4 or x > 2.4 
	or theta < -math.rad(12)
	or theta > math.rad(12)
	then
		return 0	-- invalid state means we've failed
	end

	local xIndex = bin(x, -.8, .8, xBins)
	local dxdtIndex = bin(dx_dt, -.5, .5, dxdtBins)

	-- theta is nonlinear...
	assert(thetaBins == 6)
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

	local dthetadtIndex = bin(dtheta_dt, -math.rad(50), math.rad(50), dthetadtBins)

	return 
		xIndex + xBins * (
		dxdtIndex + dxdtBins * (
		thetaIndex + thetaBins * (
		dthetadtIndex)))
		+ 1 	-- +1 for lua
end

function CartController:getAction(x, dx_dt, theta, dtheta_dt, reinforcement)
	self.prevState = self.curState
	self.prevAction = self.curAction
	self.curState = getState(x, dx_dt, theta, dtheta_dt)
	
	local predictedValue
	if self.prevAction ~= 0 then
		if self.curState == 0 then
			predictedValue = 0
		elseif self.qnn.nn.w[1][1][self.curState] <= self.qnn.nn.w[1][2][self.curState] then
			predictedValue = self.qnn.nn.w[1][2][self.curState]
		else
			predictedValue = self.qnn.nn.w[1][1][self.curState]
		end
		
		local alpha = .5
		local gamma = .999
		self.qnn.nn.w[1][self.prevAction][self.prevState]
		= self.qnn.nn.w[1][self.prevAction][self.prevState]
		 + alpha * (reinforcement + gamma * predictedValue - self.qnn.nn.w[1][self.prevAction][self.prevState]) 
	end

	self.curAction = 0
	if self.curState ~= 0 then
		local beta = 0
		if self.qnn.nn.w[1][1][self.curState] < self.qnn.nn.w[1][2][self.curState] then
			self.curAction = 2
		else
			self.curAction = 1
		end
	end

	return self.curAction
end

local Cart = class()
Cart.maxIterations = 100000
Cart.maxFailures = 3000
function Cart:init()
	self.bestIteration = 0
	self.bestFailures = 0
	self.controller = CartController()
	self:reset()
end
function Cart:reset()
	self.x = 0
	self.dx_dt = 0
	self.theta = (math.random() * 2 - 1) * math.rad(6)
	self.dtheta_dt = 0
	self.iteration = 0
	self.failures = 0
	self.controller:reset()
end
function Cart:step()
	if self.iteration < self.maxIterations
	and self.failures < self.maxFailures then 
		self.iteration = self.iteration + 1
		local action = self.controller:getAction(self.x, self.dx_dt, self.theta, self.dtheta_dt, 0)

		local failed = self:simulate(action)

		if failed then
			self.failures = self.failures + 1
			printf('Trial %d was %d steps.', self.failures, self.iteration)
			if self.iteration > self.bestIteration then
				self.bestIteration = self.iteration
				self.bestFailures = self.failures
			end
			-- reinforce
			self.controller:getAction(self.x, self.dx_dt, self.theta, self.dtheta_dt, -1)
			self.controller:reset()
			self:reset()
		end
	else
		if self.failures >= self.maxFailures then
			printf("Pole not balanced. Stopping after %d failures.", self.failures)
			printf("High water mark: %d steps in trial %d.", self.bestIteration, self.bestFailures)
		else
			printf("Pole balanced successfully for at least %d steps in trial %d.", self.iteration - 1, self.failures + 1)
		end
		os.exit()
	end
end
function Cart:simulate(action)
	local gravity = 9.8
	local massCart = 1
	local massPole = .1
	local totalMass = massPole + massCart
	local length = .5
	local poleMassLength = massPole * length
	local forceMag = 10
	local dt = .02

	local force = ({-forceMag, forceMag})[action] or 0
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

openglapp:run{
	init = function()
	end,
	update = function()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)

		cart:step()

		local width, height = openglapp:size()
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
	end,
}
