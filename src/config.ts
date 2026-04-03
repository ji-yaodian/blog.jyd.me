export const SITE = {
  website: "https://blog.jyd.me/",
  author: "jyd",
  profile: "https://github.com/ji-yaodian",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 10,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true,
  editPost: {
    enabled: false,
    url: "",
  },
  dynamicOgImage: true,
  lang: "zh-CN",
  timezone: "Asia/Shanghai",
} as const;
