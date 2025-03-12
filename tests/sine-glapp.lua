#! /usr/bin/env luajit
--[[
ok so after complex_mul's sucess, now to try to piecewise approximate a sine wave
--]]
local ffi = require 'ffi'
local gl = require 'gl'
local GLSceneObject = require 'gl.sceneobject'
local ANN = require 'neuralnet.ann'

local App = require 'imguiapp.withorbit'()
local f = math.sin
local xmin = 0
local xmax = 2 * math.pi
local n = 1000

function App:initGL()
	App.super.initGL(self)
	nn = ANN(1,10,1)

--[[
	self.view.ortho = true
	self.view.orthoSize = 3
	local xc = .5 * (xmin + xmax)	
	self.view.orbit:set(xc,0,0)
	self.view.pos:set(xc,0,10)
--]]

	local shader = require 'gl.program'{
		version = 'latest',
		precision = 'best',
		vertexCode = [[
in vec2 vertex;
in mat4 mvProjMat;
void main() {
	gl_Position = mvProjMat * vec4(vertex, 0., 1.);
}
]],
		fragmentCode = [[
out vec4 fragColor;
uniform vec3 color;
void main() {
	fragColor = vec4(color, 1.);
}
]],
	}

	self.ySceneObj = GLSceneObject{
		program = shader,
		geometry = {
			mode = gl.GL_LINE_STRIP,
			count = n,
		},
		uniforms = {
			color = {1,0,0},
		},
		vertexes = {
			useVec = true,
			dim = 2,
			count = n,
		},
	}

	self.fSceneObj = GLSceneObject{
		program = shader,
		geometry = {
			mode = gl.GL_LINE_STRIP,
			count = n,
		},
		uniforms = {
			color = {0,1,0},
		},
		vertexes = {
			useVec = true,
			dim = 2,
			count = n,
		},
	}
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

	self.ySceneObj.uniforms.mvProjMat = self.view.mvProjMat.ptr
	local vec = self.ySceneObj:beginUpdate()
	for i=0,n-1 do
		local x = (i+.5)/n * (xmax - xmin) + xmin
		nn.input[1] = x
		nn:feedForward()
		local y = nn.output[1]
		vec.v[i].x = x
		vec.v[i].y = y
	end
	self.ySceneObj:endUpdate()

	self.fSceneObj.uniforms.mvProjMat = self.view.mvProjMat.ptr
	local vec = self.fSceneObj:beginUpdate()
	for i=0,n-1 do
		local x = (i+.5)/n * (xmax - xmin) + xmin
		vec.v[i].x = x
		vec.v[i].y = f(x)
	end
	self.fSceneObj:endUpdate()

	App.super.update(self)
end
return App():run()
