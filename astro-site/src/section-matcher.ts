import { sectionByPath } from "./navigation.mjs";
export interface HeliaSectionLink {
  label: string;
  href: string;
  match?: string;
  sidebar?: false;
}
export function matchSection<T extends HeliaSectionLink>(
  sections: readonly T[],
  pathname: string,
  base: string,
): T | undefined {
  const path = pathname.replace(/\/?$/, "/");
  const assigned = sectionByPath[path];
  if (assigned) return sections.find((section) => section.href === assigned);
  const fallback = path.startsWith(`${base.replace(/\/$/, "")}/detection-`)
    ? "/sleepkit/tasks/"
    : path.startsWith(`${base.replace(/\/$/, "")}/evidence/`)
      ? "/sleepkit/guides/"
      : undefined;
  if (fallback) return sections.find((section) => section.href === fallback);
  return sections
    .filter((section) => {
      const prefix = (section.match ?? section.href).replace(/\/?$/, "/");
      return prefix === base.replace(/\/?$/, "/")
        ? path === prefix
        : path.startsWith(prefix);
    })
    .sort((a, b) => (b.match ?? b.href).length - (a.match ?? a.href).length)[0];
}
