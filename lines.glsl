
---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

uniform mat4 normal_mat;
uniform vec4 lineColor;

void main (void){
    gl_FragColor = lineColor;
}