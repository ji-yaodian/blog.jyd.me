import IconGitHub from "@/assets/icons/IconGitHub.svg";
import IconMail from "@/assets/icons/IconMail.svg";
import { translateFor } from "@/i18n/utils";
import type { Props } from "astro";

type Translator = ReturnType<typeof translateFor>;

interface Social {
  name: string;
  href: string;
  linkTitle: (t: Translator) => string;
  icon: (_props: Props) => Element;
}

export const SOCIALS: Social[] = [
  {
    name: "Github",
    href: "https://github.com/ji-yaodian",
    linkTitle: (t: Translator) => t("socials.github"),
    icon: IconGitHub,
  },
] as const;

export const SHARE_LINKS: Social[] = [] as const;
