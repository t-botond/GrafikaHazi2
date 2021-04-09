#include "framework.h"

void printVec3(const vec3& v, const char* s = "vektor:") {
	printf("%s (%.4f; %.4f; %.4f)\n", s, v.x, v.y, v.z);
}

void SampleMirror(const vec3& N, const vec3& inDir, vec3& outDir) {
	outDir = inDir - N * dot(N, inDir) * 2.0f;
}

const float epsilon = 0.0001f;

const float dodeka_vertices[] = {
	0.0f,		0.618f,		1.618f,		//0
	0.0f,		-0.618f,	1.618f,		//1
	0.0f,		-0.618f,	-1.618f,	//2
	0.0f,		0.618f,		-1.618f,	//3
	1.618f,		0.0f,		0.618f,		//4
	-1.618f,	0.0f,		0.618f,
	-1.618f,	0.0f,		-0.618f,	//6
	1.618f,		0.0f,		-0.618f,
	0.618f,		1.618f,		0.0f,		//8
	-0.618f,	1.618f,		0.0f,
	-0.618f,	-1.618f,	0.0f,		//10
	0.618f,		-1.618f,	0.0f,
	1.0f,		1.0f,		1.0f,		//12
	-1.0f,		1.0f,		1.0f,
	-1.0f,		-1.0f,		1.0f,		//14
	1.0f,		-1.0f,		1.0f,
	1.0f,		-1.0f,		-1.0f,		//16
	1.0f,		1.0f,		-1.0f,
	-1.0f,		1.0f,		-1.0f,
	-1.0f,		-1.0f,		-1.0f		//19
};

//0-tól indexelve
const size_t dodeka_sides[] = {
	0,	1,	15,	5,	13,
	0,	12,	8,	9,	13,
	0,	13,	5,	14,	2,
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

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
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

bool pTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
	a = a - p;
	b = b - p;
	c = c - p;
	float szog = acosf(dot(a, b) / (length(a) * length(b)));
	szog = szog + acosf(dot(a, c) / (length(a) * length(c)));
	szog = szog + acosf(dot(c, b) / (length(b) * length(c)));
	return (2 * M_PI - epsilon * 10 < szog && 2 * M_PI + epsilon * 10 > szog);
}
struct oTriangle :public Intersectable {
	const vec3 a,b,c; 
	oTriangle(const vec3& _a, const vec3& _b, const vec3& _c, Material* _mat):a(_a), b(_b), c(_c) {
		material = _mat;
	}
	Hit intersect(const Ray& ray) {
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
		SampleMirror(n, normalize(ray.dir), hit.normal);
		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

struct Dodeka{
	Material* material;
	float vertices[20*3];
	Dodeka(const vec3& eltolas, Material* _material) {
		material = _material; 
		for (size_t i = 0; i < 20; ++i) {
			vertices[(i * 3) + 0] = dodeka_vertices[(i * 3) + 0] + eltolas.x;
			vertices[(i * 3) + 1] = dodeka_vertices[(i * 3) + 1] + eltolas.y;
			vertices[(i * 3) + 2] = dodeka_vertices[(i * 3) + 2] + eltolas.z;
		}
	}

	void build(std::vector<Intersectable*>& objects) {
		vec3 elvart1 = vec3(0.0f, 0.618f, 1.618f);
		vec3 elvart2 = vec3(1.0f, 1.0f, 1.0f);
		vec3 elvart3 = vec3(0.618f, 1.618f, 0.0f);


		vec3* kapott = getTriangleAt(2, 0);
		printVec3(elvart1, "Elvart1:");
		printVec3(kapott[0], "Kapott1:");
		printVec3(elvart2, "Elvart2:");
		printVec3(kapott[1], "Kapott2:");
		printVec3(elvart3, "Elvart3:");
		printVec3(kapott[2], "Kapott3:");





	}
	vec3* getTriangleAt(const size_t side, const size_t idx) {
		vec3* ret = new vec3[3];
		size_t v1 = dodeka_sides[side];
		size_t v2 = dodeka_sides[side + 1 + idx];
		size_t v3 = dodeka_sides[side + 2 + idx];
		ret[0] = vec3(vertices[v1 * 3], vertices[(v1 * 3) + 1], vertices[(v1 * 3) + 2]);
		ret[1] = vec3(vertices[v2 * 3], vertices[(v2 * 3) + 1], vertices[(v2 * 3) + 2]);
		ret[2] = vec3(vertices[v3 * 3], vertices[(v3 * 3) + 1], vertices[(v3 * 3) + 2]);
		return ret;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
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
};
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, -5), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		//vec3 lightDirection(0, 0, 0), Le(0, 0, -10);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.17f, 0.35f, 1.5f);
		vec3 ks(3.1f, 2.7f, 1.9f);

		Material* material = new Material(kd, ks, 100);
		kd=vec3(0.3f, 0.2f, 0.1f), ks=vec3(2, 2, 2);
		Material* material2 = new Material(kd, ks, 100);
		Dodeka d = Dodeka(vec3(0, 0, 0), material2);
		d.build(objects);
		
		objects.push_back(new oTriangle(vec3(0.0f, 0.618f, 1.618f), vec3(1.0f, 1.0f, 1.0f), vec3(0.618f, 1.618f, 0.0f), material));
		objects.push_back(new oTriangle(vec3(0.0f, 0.618f, 1.618f), vec3(0.618f, 1.618f, 0.0f), vec3(-0.618f, 1.618f, 0.0f), material));
		objects.push_back(new oTriangle(vec3(0.0f, 0.618f, 1.618f), vec3(-0.618f, 1.618f, 0.0f), vec3(-1.0f, 1.0f, 1.0f), material));



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
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
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
		return outRadiance;
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

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'w') {

	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}