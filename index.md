---
layout: default
---

<section class="hero">
  <h1>Tensor Slayer</h1>
  <p>Practical experiments in model behavior, interpretability, and binary-level interventions. Built for fast iteration and clear reporting.</p>
</section>

<h2 class="section-title">Recent posts</h2>
<ul class="posts-grid">
  {% for post in site.posts %}
  <li class="post-card">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <div class="post-meta">{{ post.date | date: "%Y-%m-%d" }}</div>
    <div class="post-excerpt">{{ post.excerpt | strip_html | strip_newlines }}</div>
  </li>
  {% endfor %}
</ul>

<section class="callout">
  Thoughts = Projects. mostly left half assed soon after attaining first dopamine hit.
</section>

## Recent Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.excerpt | strip_html | strip_newlines }}
{% endfor %}
