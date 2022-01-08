
---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_norm;
attribute vec2  v_texc;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform vec3 light_direction;

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 texcoord;
varying vec4 light_vec;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    vertex_pos = pos;
    texcoord = v_texc;
    light_vec = vec4(light_direction, 0.0);
    normal_vec = vec4(v_norm, 0.0);
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 texcoord;
varying vec4 light_vec;

uniform bool useDarkTexture;
uniform mat4 normal_mat;
uniform sampler2D theTexture;
uniform sampler2D secondTexture;

void main (void){

    vec4 v_normal = normalize(normal_mat * normal_vec);
    vec4 v_light = normalize(normal_mat * light_vec);
    float theta = dot(v_normal, v_light);
    if (useDarkTexture) {
        const float contrast = 0.4;
        const float sharpness = 6.0;
        const float offset = 0.85;
        float b1 = min(abs(theta) * contrast + (1.0-contrast), 1.0);
        float b2 = clamp(offset+sharpness*theta, 0.0, 1.0);
        vec4 blend = (b2) * texture2D(theTexture, texcoord) + 
               (1.0 - b2) * texture2D(secondTexture, texcoord);
        gl_FragColor = vec4(b1, b1, b1, 1.0) * blend;
    } else {
        const float contrast = 0.5;
        theta = clamp(theta, 0.0, 1.0) * contrast + (1.0-contrast);
        gl_FragColor = vec4(theta, theta, theta, 1.0) * texture2D(theTexture, texcoord);
    }
}