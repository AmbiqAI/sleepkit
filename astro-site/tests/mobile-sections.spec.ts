import { test, expect } from '@playwright/test';

const base = '/sleepkit';
for (const width of [390, 835]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`section dropdown scopes navigation at ${width} in ${colorScheme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 844 });
      await page.emulateMedia({ colorScheme });
      await page.goto(`${base}/guides/`);
      const dropdown = page.locator('[data-helia-section-dropdown]');
      const summary = dropdown.locator('summary');
      await expect(summary).toContainText('User guide');
      await summary.focus();
      await page.keyboard.press('Enter');
      await expect(dropdown).toHaveAttribute('open', '');
      await page.keyboard.press('Escape');
      await expect(dropdown).not.toHaveAttribute('open', '');
      await expect(summary).toBeFocused();
      const menu = page.getByRole('button', { name: 'Open navigation', exact: true });
      const sidebar = page.locator('#starlight__sidebar');
      await menu.click();
      await expect(sidebar.locator('[data-helia-sidebar-heading]')).toHaveText('User guide');
      const guideLinks = await sidebar.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      await menu.click();
      await summary.click();
      await page.getByRole('navigation', { name: 'Choose section' }).getByRole('link', { name: 'Getting started', exact: true }).click();
      await expect(summary).toContainText('Getting started');
      await menu.click();
      await expect(sidebar.locator('[data-helia-sidebar-heading]')).toHaveText('Getting started');
      const startLinks = await sidebar.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      expect(startLinks.length).toBeGreaterThan(0);
      expect(startLinks).not.toEqual(guideLinks);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: `/tmp/sleepkit-rollout-${width}-${colorScheme}.png` });
    });
  }
}

test('desktop retains direct section navigation', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}/`);
  await expect(page.locator('[data-helia-section-dropdown]')).toBeHidden();
  await expect(page.getByRole('navigation', { name: 'Primary', exact: true }).getByRole('link', { name: 'Getting started', exact: true })).toBeVisible();
});

for (const colorScheme of ['light', 'dark'] as const) {
  test(`tabbed terminals keep commands and copy controls in ${colorScheme}`, async ({ page }) => {
    await page.setViewportSize({ width: 835, height: 890 });
    await page.emulateMedia({ colorScheme });
    await page.goto(`${base}/quickstart/`);
    const terminal = page.locator('[role="tabpanel"] .frame.is-terminal:visible').first();
    await expect(terminal).toBeVisible();
    await expect(terminal.locator('.header')).toBeHidden();
    await expect(terminal.locator('pre')).not.toHaveText('');
    await terminal.hover();
    await expect(terminal.getByRole('button', { name: 'Copy to clipboard' })).toBeVisible();
    await terminal.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `/tmp/sleepkit-rollout-terminal-${colorScheme}.png` });
  });
}
