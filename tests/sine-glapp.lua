#! /usr/bin/env luajit
--[[
ok so after complex_mul's sucess, now to try to piecewise approximate a sine wave
--]]
local gl = require 'gl'
local ANN = require 'neuralnet.ann'

local App = require 'imguiapp.withorbit'()

local f = math.sin
local xmin = 0
local xmax = 2 * math.pi
local n = 1000


local function showCurrentResult(args)
end

function App:initGL()
	App.super.initGL(self)
	nn = ANN(1,10,1)

	self.view.ortho = true
	self.view.orthoSize = 3
	local xc = .5 * (xmin + xmax)	
	self.view.orbit:set(xc,0,0)
	self.view.pos:set(xc,0,10)
end

function App:update()
	gl.glClear(gl.GL_COLOR_BUFFER_BIT)

	-- right at 100,000 the max error seem to drop to .01.  then at 10,000,000 it drops to .0001
	local t = math.random() * (xmax - xmin) + xmin
	nn.input[1] = t
	nn:feedForward()
	nn.desired[1] = f(t)
	local err = nn:calcError()
	--results:insert(err)
	nn:backPropagate(.1)

	gl.glColor3f(1,0,0)	
	gl.glBegin(gl.GL_LINE_STRIP)
	for i=1,n do
		local x = (i-.5)/n * (xmax - xmin) + xmin
		nn.input[1] = x
		nn:feedForward()
		local y = nn.output[1]
		gl.glVertex2f(x,y)
	end
	gl.glEnd()

	gl.glColor3f(0,1,0)	
	gl.glBegin(gl.GL_LINE_STRIP)
	for i=1,n do
		local x = (i-.5)/n * (xmax - xmin) + xmin
		gl.glVertex2f(x,f(x))
	end
	gl.glEnd()

	App.super.update(self)
end
return App():run()
