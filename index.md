---
layout: home
---

# Welcome to Tensor Slayer's Blog

Thoughts = Projects. mostly left half assed soon after attaining first dopamine hit.

## Recent Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.excerpt | strip_html | strip_newlines }}
{% endfor %}
