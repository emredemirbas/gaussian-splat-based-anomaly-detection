#version 430 core

uniform vec3 bbox_color;

out vec4 FragColor;

void main()
{
    FragColor = vec4(bbox_color, 1.0);
}
