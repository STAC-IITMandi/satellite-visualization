
---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_norm;
attribute vec2  v_texc;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 texcoord;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    vertex_pos = pos;
    texcoord = v_texc;

    normal_vec = vec4(v_norm,0.0);
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 texcoord;

uniform mat4 normal_mat;
uniform sampler2D theTexture;

void main (void){
    vec4 v_normal = normalize( normal_mat * normal_vec ) ;
    vec4 v_light = normalize( vec4(0,0,0,1) - vertex_pos );
    float theta = clamp(dot(v_normal, v_light), 0.0, 1.0) * 0.5 + 0.5;
    // gl_FragColor = texture2D(theTexture, texcoord);
    gl_FragColor = vec4(theta, theta, theta, 1.0) * texture2D(theTexture, texcoord);
}