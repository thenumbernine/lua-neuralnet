#! /usr/bin/env luajit
--[[
ok so after complex_mul's sucess, now to try to piecewise approximate a sine wave
--]]
local ffi = require 'ffi'
local gl = require 'gl'
local ANN = require 'neuralnet.ann'

local App = require 'imgui.appwithorbit'()
local f = math.sin
local xmin = 0
local xmax = 2 * math.pi
local n = 1000

function App:initGL()
	App.super.initGL(self)
	nn = ANN(1,8,1)
	nn.useBatch = 10
	nn:setActivation'poorQuadraticTanh'
	nn:setActivationDeriv'poorQuadraticTanhDeriv'

-- [[
	self.view.ortho = true
	self.view.orthoSize = 3
	local xc = .5 * (xmin + xmax)	
	self.view.orbit:set(xc,0,0)
	self.view.pos:set(xc,0,10)
--]]

	self.lineSceneObj = require 'gl.sceneobject'{
		program = {
			version = 'latest',
			precision = 'best',
			vertexCode = [[
in vec2 vertex;
uniform mat4 mvProjMat;
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
		},
		geometry = {
			mode = gl.GL_LINE_STRIP,
		},
		uniforms = {
			color = {1,0,0},
			mvProjMat = self.view.mvProjMat.ptr,
		},
		vertexes = {
			useVec = true,
			dim = 2,
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
	nn:backPropagate(.1 / nn.useBatch)

	self.lineSceneObj.uniforms.color = {1,0,0}
	local vec = self.lineSceneObj:beginUpdate()
	for i=0,n-1 do
		local x = (i+.5)/n * (xmax - xmin) + xmin
		nn.input[1] = x
		nn:feedForward()
		local y = nn.output[1]
		vec:emplace_back():set(x, y)
	end
	self.lineSceneObj:endUpdate()

	self.lineSceneObj.uniforms.color = {0,1,0}
	local vec = self.lineSceneObj:beginUpdate()
	for i=0,n-1 do
		local x = (i+.5)/n * (xmax - xmin) + xmin
		vec:emplace_back():set(x, f(x))
	end
	self.lineSceneObj:endUpdate()

	App.super.update(self)
end

return App():run()
