#include "framework.h"
const int MAXDEPTH = 5;
bool forgas = false;
const float epsilon = 0.0001f;
const float dodeka_vertices[] = {
	0.0f,		0.618f,		1.618f,
	0.0f,		-0.618f,	1.618f,
	0.0f,		-0.618f,	-1.618f,
	0.0f,		0.618f,		-1.618f,
	1.618f,		0.0f,		0.618f,
	-1.618f,	0.0f,		0.618f,
	-1.618f,	0.0f,		-0.618f,
	1.618f,		0.0f,		-0.618f,
	0.618f,		1.618f,		0.0f,
	-0.618f,	1.618f,		0.0f,
	-0.618f,	-1.618f,	0.0f,
	0.618f,		-1.618f,	0.0f,
	1.0f,		1.0f,		1.0f,
	-1.0f,		1.0f,		1.0f,
	-1.0f,		-1.0f,		1.0f,
	1.0f,		-1.0f,		1.0f,
	1.0f,		-1.0f,		-1.0f,
	1.0f,		1.0f,		-1.0f,
	-1.0f,		1.0f,		-1.0f,
	-1.0f,		-1.0f,		-1.0f
};
const size_t dodeka_sides[] = {
	0,	1,	15,	4,	12,
	0,	12,	8,	9,	13,
	0,	13,	5,	14,	1,
	1,	14,	10,	11,	15,
	2,	3,	17,	7,	16,
	2,	16,	11,	10,	19,
	2,	19,	6,	18,	3,
	18,	9,	8,	17,	3,
	15,	11,	16,	7,	4,
	4,	7,	17,	8,	12,
	13,	9,	18,	6,	5,
	5,	6,	19,	10,	14
};
enum MaterialType { ROUGH, REFLECTIVE, MIRROR };

inline void printVec3(const vec3& v, const char* s = "vektor:", const char* nl = "\n") {
	printf("%s (%.4f; %.4f; %.4f)%s", s, v.x, v.y, v.z, nl);
}
inline vec3 normalize2(const vec3& v, float targetLength = 1.0f) { return v * (targetLength / length(v)); }
inline vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
inline vec3 operator*=(vec3 a, vec3 b) {
	return a * b;
}
vec3 reflect(const vec3& dir, const vec3& n) {
	return dir - n * dot(n, dir) * 2.0f;
}

//Forrás: Az elõadás videókból
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) :type(t) {}
};
//Forrás: Az elõadás videókból
struct RoughMaterial : public Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) :Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};
//Forrás: Az elõadás videókból
struct ReflectiveMaterial : public Material {
	ReflectiveMaterial(vec3 n, vec3 kappa):Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};
struct Tukor :public Material {
	Tukor() :Material(MIRROR) {}
};

//Forrás: Az elõadás videókból
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};
//Forrás: Az elõadás videókból
struct Ray {
	vec3 start, dir, weight;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); weight = vec3(1, 1, 1); }
};
//Forrás: Az elõadás videókból
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};
class oTriangle :public Intersectable {
protected:
	const vec3 a, b, c;
public:
	oTriangle(const vec3& _a, const vec3& _b, const vec3& _c, Material* _mat) :a(_a), b(_b), c(_c) {
		material = _mat;
	}
	virtual Hit intersect(const Ray& ray) {
		Hit hit;
		const vec3 n = cross(c - a, b - a);
		const float t = (dot((a - ray.start), n)) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;
		if (dot(cross((c - a), (p - a)), n) <= 0) return hit;
		if (dot(cross((b - c), (p - c)), n) <= 0) return hit;
		if (dot(cross((a - b), (p - b)), n) <= 0) return hit;
		hit.t = t;
		hit.position = p;
		hit.normal = normalize(n);
		hit.material = material;
		return hit;
	}
};
struct Sphere : public Intersectable {
	vec3 center;
	float radius;
	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};
class KozepsoTargy : public Intersectable {
protected:
	vec3 params;
public:
	KozepsoTargy(vec3 _params, Material* _material) :params(_params) {
		material = _material;
		//printf("%.4f", log(exp(7)));
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		float a = params.x * ray.dir.x * ray.dir.x + params.y * ray.dir.y * ray.dir.y;
		float b = 2.0f * params.x * ray.start.x * ray.dir.x + 2.0f * params.y * ray.start.y * ray.dir.y - params.z * ray.dir.z;
		float c = params.x * ray.start.x * ray.start.x + params.y * ray.start.y * ray.start.y - params.z * ray.start.z;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		float t = (t2 > 0) ? t2 : t1;
		vec3 p = ray.start + ray.dir * t;
		if (length(vec3() - p) > 0.3f) {
			return hit;
		}
		hit.t = t;
		hit.position = p;
		hit.normal = normalize(vec3(
			2.0f * params.x * p.x * expf(params.x * p.x * p.x + params.y * p.y * p.y - params.z * p.z), 
			2.0f * params.y * p.y * expf(params.x * p.x * p.x + params.y * p.y * p.y - params.z * p.z), 
			-params.z * expf(params.x * p.x * p.x + params.y * p.y * p.y - params.z * p.z)));
		hit.material = material;
		return hit;
	}
};

class sTr :public oTriangle {
protected:
	const vec3 xa, xb, xc, xd, xe;
public:
	sTr(const vec3& _a, const vec3& _b, const vec3& _c, Material* _mat, const vec3& _x, const vec3& _y, const vec3& _z, const vec3& _zd, const vec3& _ze) :oTriangle(_a, _b, _c, _mat), xa(_x), xb(_y), xc(_z), xd(_zd), xe(_ze) {}
	Hit intersect(const Ray& ray) {
		Hit hit;
		const vec3 n = cross(c - a, b - a);
		const float t = (dot((a - ray.start), n)) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;
		if (dot(cross((c - a), (p - a)), n) <= 0) return hit;
		if (dot(cross((b - c), (p - c)), n) <= 0) return hit;
		if (dot(cross((a - b), (p - b)), n) <= 0) return hit;
		if (onTriangle(ray, xa, xb, xc) || onTriangle(ray, xa, xc, xd) || onTriangle(ray, xa, xd, xe)) return hit;
		if (onTriangle(ray, xb, xc, xd) || onTriangle(ray, xb, xd, xe) || onTriangle(ray, xb, xe, xa)) return hit;
		if (onTriangle(ray, xc, xd, xe) || onTriangle(ray, xc, xe, xa) || onTriangle(ray, xc, xa, xb)) return hit;
		hit.t = t;
		hit.position = p;
		hit.normal = normalize(n);
		hit.material = material;
		return hit;
	}
	bool onTriangle(const Ray& ray, vec3 va, vec3 vb, vec3 vc) {
		const vec3 n = cross(vc - va, vb - va);
		const float t = (dot((va - ray.start), n)) / dot(ray.dir, n);
		if (t < 0) return false;
		vec3 p = ray.start + ray.dir * t;
		if (dot(cross((vc - va), (p - va)), n) <= 0) return false;
		if (dot(cross((vb - vc), (p - vc)), n) <= 0) return false;
		if (dot(cross((va - vb), (p - vb)), n) <= 0) return false;
		return true;
	}
};
class Dodeka {
	Material* oldal;
	float vertices[20 * 3];
	Material* tukor;
public:
	Dodeka(const vec3& eltolas, Material* _material) {
		oldal = _material;
		for (size_t i = 0; i < 20; ++i) {
			vertices[(i * 3) + 0] = dodeka_vertices[(i * 3) + 0] + eltolas.x;
			vertices[(i * 3) + 1] = dodeka_vertices[(i * 3) + 1] + eltolas.y;
			vertices[(i * 3) + 2] = dodeka_vertices[(i * 3) + 2] + eltolas.z;
		}
		//tukor = new ReflectiveMaterial(vec3(0.17f, 0.35f,1.5f), vec3(3.1f, 2.7f,1.9f));
		tukor = new Tukor();

	}
	void build(std::vector<Intersectable*>& objects) {
		for (size_t side = 0; side < 12; ++side) {
			vec3* v = getSide(side);
			vec3 a = v[0] + normalize2((v[3] - v[0]) + (v[2] - v[0]), 0.12361f);
			vec3 b = v[1] + normalize2((v[4] - v[1]) + (v[3] - v[1]), 0.12361f);
			vec3 c = v[2] + normalize2((v[0] - v[2]) + (v[4] - v[2]), 0.12361f);
			vec3 d = v[3] + normalize2((v[0] - v[3]) + (v[1] - v[3]), 0.12361f);
			vec3 e = v[4] + normalize2((v[1] - v[4]) + (v[2] - v[4]), 0.12361f);
			objects.push_back(new sTr(v[0], v[1], v[2], oldal, a, b, c, d, e));
			objects.push_back(new sTr(v[0], v[2], v[3], oldal, a, c, d, b, e));
			objects.push_back(new sTr(v[0], v[3], v[4], oldal, a, d, e, b, c));

			objects.push_back(new oTriangle(v[0], v[1], v[2], tukor));
			objects.push_back(new oTriangle(v[0], v[2], v[3], tukor));
			objects.push_back(new oTriangle(v[0], v[3], v[4], tukor));
			delete[] v;
		}
	}
	vec3* getSide(const size_t side) {
		vec3* ret = new vec3[5];
		for (size_t i = 0; i < 5; ++i) {
			size_t v = dodeka_sides[(side * 5) + i];
			ret[i] = vec3(vertices[v * 3], vertices[(v * 3) + 1], vertices[(v * 3) + 2]);
		}
		return ret;
	}
};

//Forrás: Az elõadás videókból
class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov=-1) {
		if (_fov > 0)fov = _fov;
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up);//, 45 * M_PI / 180
	}
};
//Forrás: Az elõadás videókból
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};
//Forrás: Az elõadás videókból
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0.3f, -0.8f, -0.8f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(3, 3, 3);
		lights.push_back(new Light(lightDirection, Le));
		Material* sargas = new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), vec3(2,2,2), 100);
		Material* arany = new ReflectiveMaterial(vec3(0.17f,0.35f,1.5f), vec3(3.1f,2.7f,1.9f));
		Dodeka d = Dodeka(vec3(), sargas);
		d.build(objects);
		//objects.push_back(new Sphere(vec3(), 0.30f, arany));
		objects.push_back(new KozepsoTargy(vec3(0.3f,2.1f,0.1f),arany));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}

	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > MAXDEPTH) return La;
		Hit hit = firstIntersect(ray);

		if (hit.t < 0) return La;
		vec3 outRadiance(0,0,0);
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F= hit.material->F0 + (vec3(1,1,1) - hit.material->F0) * pow(1- cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), ++depth) * F;
		}
		if (hit.material->type == MIRROR) {
			ray.weight *= hit.material->F0 + (vec3(1, 1, 1) - hit.material->F0) * pow(dot(-ray.dir, hit.normal), 5);
			ray.start = hit.position + hit.normal + epsilon;
			ray.dir = reflect(ray.dir, hit.normal);
			outRadiance = outRadiance + trace(ray, ++depth);
		}
		return outRadiance;
	}
	void Animate(float dt) {
		camera.Animate(dt);
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";
//Forrás: Az elõadás videókból
class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
FullScreenTexturedQuad* fullScreenTexturedQuad;

//Forrás: Az elõadás videókból
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
//Forrás: Az elõadás videókból
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		forgas = !forgas;
	}
}
void onKeyboardUp(unsigned char key, int pX, int pY) {

}
void onMouse(int button, int state, int pX, int pY) {
}
void onMouseMotion(int pX, int pY) {
}
void onIdle() {
	if (forgas) {
		scene.Animate(0.1f);
		std::vector<vec4> image(windowWidth * windowHeight);
		scene.render(image);
		delete fullScreenTexturedQuad;
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		glutPostRedisplay();
	}
}