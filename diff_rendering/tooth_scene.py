import math
import mitsuba as mi
from mitsuba import ScalarTransform4f as T

def create_scene(spp=128, res=256):
	return {
		'type': 'scene',
		'sensor': {
			'type': 'orthographic',
			'to_world': T.scale((5,)*3).look_at(origin=[0,1,0], target=[0,0,0], up=[0,0,1]),
			'sampler': {
				'type': 'independent',
				'sample_count': spp
			},
			'film': {
				'type': 'hdrfilm',
				'width': res,
				'height': res,
				'rfilter': {
					'type': 'box'
				},
				'pixel_format': 'rgb',
				'component_format': 'float32',
				'sample_border': True
			}
		},
		'integrator': {
			'type': 'path',
			'max_depth': 3
		},
		'veryblack': {
			'type': 'diffuse',
			'reflectance': { 'type': 'spectrum', 'value': 0 }
		},
		'black': {
			'type': 'diffuse',
			'reflectance': { 'type': 'spectrum', 'value': 0.1 }
		},
		'white': {
			'type': 'diffuse',
			'reflectance': { 'type': 'spectrum', 'value': 1 }
		},
		'env': {
			'type': 'constant',
			'radiance': { 'type': 'spectrum', 'value': 100 }
		},
		'floor': {
			'type': 'obj',
			'filename': 'floor.obj',
			'bsdf': { 'type': 'ref', 'id': 'white'}
		},
		'walls': {
			'type': 'obj',
			'filename': 'walls.obj',
			'bsdf': { 'type': 'ref', 'id': 'black'}
		},
		'underfloor': {
			'type': 'rectangle',
			'bsdf': { 'type': 'ref', 'id': 'veryblack'},
			'to_world': T.scale((5,)*3).look_at(origin=[0,-1,0], target=[0,0,0], up=[0,0,1])
		}
	}