# Hugo → Astro 博客迁移方案

## Context

当前博客 blog.jyd.me 使用 Hugo + PaperMod 主题，部署在 GitHub Pages。用户希望迁移到 Astro 框架，使用现成主题，部署到 Vercel，保留所有核心功能（中英双语、Giscus 评论、Google Analytics、搜索）。在当前仓库新建分支完成迁移。

## 关键决策

- **主题**: AstroPaper i18n（`yousef8/astro-paper-i18n`）— AstroPaper v5 的 i18n fork，内建多语言路由、Pagefind 搜索、暗亮切换
- **部署**: Vercel（静态站点，零配置）
- **仓库策略**: 新建 `feat/astro-migration` 分支，完成后合并到 main

---

## Phase 1: 项目初始化

### 1.1 创建迁移分支并清理 Hugo 文件
```bash
git checkout -b feat/astro-migration
```

需要移除的 Hugo 文件/目录：
- `themes/` — Git 子模块
- `.gitmodules`
- `hugo.yaml`
- `.hugo_build.lock`
- `archetypes/`
- `resources/`
- `public/` — Hugo 生成目录
- `.github/workflows/hugo.yaml` — GitHub Pages 部署
- `vercel.json` — 旧的 Hugo 版本配置

### 1.2 初始化 Astro 项目
```bash
pnpm create astro@latest --template yousef8/astro-paper-i18n . --force-overwrite
```

更新 `.gitignore`：
```
node_modules/
dist/
.astro/
```

### 关键文件
- `astro.config.mjs` — Astro 主配置
- `src/config.ts` — 网站元数据配置
- `src/content.config.ts` — 内容集合 schema
- `package.json` — 依赖管理

---

## Phase 2: 站点配置

### 2.1 Astro i18n 配置（`astro.config.mjs`）
```javascript
i18n: {
  defaultLocale: "zh",
  locales: ["zh", "en"],
  routing: {
    prefixDefaultLocale: false,  // 中文路径无前缀，英文带 /en/
  }
}
```

### 2.2 网站元数据（`src/config.ts`）
- 站点标题: "记要点" (zh) / "JYD" (en)
- 描述: "记录技术上、生活上的各种想法、笔记、要点"
- 作者: jyd
- 社交链接: GitHub (ji-yaodian), RSS

### 2.3 UI 翻译
- `src/i18n/locales/zh.ts` — 导航、页脚、搜索提示等中文翻译
- `src/i18n/locales/en.ts` — 英文翻译（主题默认）

### 2.4 导航菜单配置
- 中文: 存档、搜索、标签、关于
- 英文: Archives、Search、Tags

### 关键配置参考
- 源: `hugo.yaml`（`/Users/jiyaodian/github/blog.jyd.me/hugo.yaml`）

---

## Phase 3: 内容迁移

### 3.1 目录结构映射

| Hugo 路径 | Astro 路径 |
|-----------|-----------|
| `content/zh/paper/*/index.md` | `src/content/blog/zh/paper-*.md` |
| `content/zh/transformers/*/index.md` | `src/content/blog/zh/transformers-*.md` |
| `content/zh/python/*.md` | `src/content/blog/zh/python-*.md` |
| `content/en/python/*.md` | `src/content/blog/en/python-*.md` |
| `content/zh/about.md` | `src/pages/about.astro`（或主题约定位置）|

### 3.2 Frontmatter 转换

变更清单：
1. `date` → `pubDate`
2. 新增 `description`（从文章首段提取）
3. 保留 `title`, `tags`, `draft`
4. 移除 Hugo 特有字段: `layout`, `url`, `categories`（AstroPaper 用 tags 代替）

转换前:
```yaml
title: "多卡训练：DP vs DDP"
date: 2023-05-20T17:45:08+08:00
tags: ['分布式', '深度学习']
categories: ['深度学习']
draft: false
```

转换后:
```yaml
title: "多卡训练：DP vs DDP"
pubDate: 2023-05-20T17:45:08+08:00
tags: ['分布式', '深度学习']
description: "数据并行和分布式数据并行训练策略的对比分析"
draft: false
```

### 3.3 Page Bundle 图片处理

Hugo Page Bundle 图片（`content/zh/paper/designEdit/` 下约 25 张图片）：
1. 移至 `public/images/<post-slug>/`
2. 更新 Markdown 中图片路径: `./fig_3.jpg` → `/images/designEdit/fig_3.jpg`

### 3.4 静态资源迁移

`static/` → `public/`:
- favicon 系列 (ico, 16x16, 32x32, android-chrome)
- `apple-touch-icon.png`
- `face.jpg`
- `site.webmanifest`
- ~~`CNAME`~~ — Vercel 不需要，删除
- `baidu_verify_*.html` — 保留（百度搜索验证）

### 关键内容文件
- `/Users/jiyaodian/github/blog.jyd.me/content/zh/` — 所有中文内容
- `/Users/jiyaodian/github/blog.jyd.me/content/en/` — 所有英文内容

---

## Phase 4: 功能集成

### 4.1 Giscus 评论组件

创建 `src/components/Giscus.astro`，使用现有配置参数：
```
data-repo="ji-yaodian/blog.jyd.me"
data-repo-id="R_kgDOJiC9Qg"
data-category="Announcements"
data-category-id="DIC_kwDOJiC9Qs4CczHJ"
data-mapping="pathname"
data-theme="preferred_color_scheme"
data-lang="zh-CN"
```

注意事项：
- 需处理 Astro View Transitions 的兼容性（监听 `astro:page-load` 事件重新加载评论）
- `data-lang` 根据当前页面语言动态设置（zh-CN / en）

在文章详情布局中引入此组件。

### 4.2 Google Analytics

在 `src/layouts/Layout.astro` 的 `<head>` 中注入 GA4 脚本：
```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-4EFJT4XVKQ"></script>
```

### 4.3 搜索

AstroPaper i18n 内建 Pagefind 搜索，开箱即用，支持中文分词。无需额外配置。

### 关键参考文件
- Giscus 配置源: `themes/PaperMod/layouts/partials/comments.html`

---

## Phase 5: 样式与细节调优

### 5.1 配置项
- 版权声明: CC BY-NC-ND 4.0 国际许可协议
- TOC（目录）: 启用
- 代码高亮: 使用 Astro 内建的 Shiki（替代 Hugo 的 Pygments）
- RSS: 验证中英文各自生成 RSS feed
- Sitemap: 验证生成多语言 sitemap

### 5.2 特殊页面
- 存档页面 (`archives`)
- 搜索页面 (`search`)
- 关于页面 (`about`)
- 标签页面 (`tags`)

---

## Phase 6: URL 兼容性

### 关键 URL 映射

| Hugo URL | Astro URL | 状态 |
|----------|-----------|------|
| `/paper/xxx/` | `/posts/paper-xxx/` 或 `/paper/xxx/` | 需确认主题路由 |
| `/transformers/xxx/` | `/posts/transformers-xxx/` | 需确认 |
| `/python/xxx/` | `/posts/python-xxx/` | 需确认 |
| `/en/python/xxx/` | `/en/posts/python-xxx/` | 需确认 |
| `/about/` | `/about/` | 直接匹配 |
| `/archives/` | `/archives/` | 直接匹配 |
| `/tags/` | `/tags/` | 直接匹配 |

如果路径发生变化，需要在 `vercel.json` 中配置 301 重定向：
```json
{
  "redirects": [
    { "source": "/paper/:slug", "destination": "/posts/paper-:slug", "permanent": true },
    { "source": "/transformers/:slug", "destination": "/posts/transformers-:slug", "permanent": true }
  ]
}
```

Giscus 评论是按 pathname 映射的，路径变化会导致评论丢失关联。优先方案是让新路径匹配旧路径。

---

## Phase 7: Vercel 部署

### 7.1 部署步骤
1. 在 Vercel Dashboard 导入 `ji-yaodian/blog.jyd.me` 仓库
2. 设置 Production Branch 为 `feat/astro-migration`（测试期间）
3. Vercel 自动检测 Astro 框架，零配置构建
4. 在 Vercel 配置自定义域名 `blog.jyd.me`
5. 更新 DNS: CNAME 从 GitHub Pages 改为 `cname.vercel-dns.com`
6. Vercel 自动处理 SSL 证书

### 7.2 测试后的最终切换
1. 先用 Vercel 预览 URL 全面测试
2. 确认无误后切换 DNS
3. 合并 `feat/astro-migration` 到 `main`
4. 更新 Vercel Production Branch 为 `main`
5. 删除旧的 `.github/workflows/hugo.yaml`

---

## Phase 8: 合并与清理

1. 合并分支到 main
2. 移除遗留的 Hugo 配置文件
3. 更新 README.md
4. 在 GitHub 仓库 Settings 中取消 GitHub Pages 配置

---

## 验证计划

### 本地验证
```bash
pnpm install
pnpm dev        # 开发模式验证
pnpm build      # 构建验证
pnpm preview    # 预览生产构建
```

### 功能验证清单
- [ ] 中文首页正常渲染
- [ ] 英文首页 (/en/) 正常渲染
- [ ] 语言切换器工作正常
- [ ] 所有文章内容完整显示
- [ ] 文章中的图片正确加载
- [ ] 代码块语法高亮正常
- [ ] TOC 目录正常展开
- [ ] Pagefind 搜索能搜到中英文内容
- [ ] Giscus 评论加载正常
- [ ] 暗/亮主题切换正常
- [ ] 标签页面显示所有标签
- [ ] 存档页面按时间排列
- [ ] 关于页面内容正确
- [ ] RSS feed 正常生成
- [ ] Sitemap 包含所有页面
- [ ] Google Analytics 脚本正确加载
- [ ] 移动端响应式布局正常
- [ ] 旧 URL 能正确访问（或被重定向）

### Vercel 部署验证
- [ ] 预览 URL 可访问
- [ ] 自定义域名 blog.jyd.me 正确解析
- [ ] HTTPS 证书正常
- [ ] 所有页面 200 状态码
- [ ] 性能评分（Lighthouse）达到合理水平
