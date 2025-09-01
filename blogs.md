---
layout: default
title: Blog
permalink: /blog/
---

<h1>Blog</h1>
<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url }}">{{ post.title }}</a> – <span>{{ post.date | date: "%B %-d, %Y" }}</span>
    <p>{{ post.excerpt }}</p>
  </li>
{% endfor %}
</ul>
