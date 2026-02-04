---
layout: home
---

<div class="index-shell">
  <section class="hero">
    <div>
      <h1>Tensor Slayer</h1>
      <p>Practical experiments in model behavior, interpretability, and binary-level interventions. Built for fast iteration and ruthless clarity.</p>
    </div>
    <div class="project-link">
      <a href="https://github.com/areu01or00/Tensor-Slayer.github.io">
        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2 .37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path>
        </svg>
        GitHub project
      </a>
    </div>
  </section>

  <section>
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
  </section>

  <section class="callout">
    Thoughts = Projects. mostly left half assed soon after attaining first dopamine hit.
  </section>
</div>

## Recent Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.excerpt | strip_html | strip_newlines }}
{% endfor %}
