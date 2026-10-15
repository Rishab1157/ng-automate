import { test, expect } from '@playwright/test';

test('passes', async ({ page }) => {
  await page.setContent('<h1 id="title">Products</h1>');
  await expect(page.locator('#title')).toHaveText('Products');
});

test('clicks a missing element', async ({ page }) => {
  await page.setContent('<button id="save">Save</button>');
  await page.locator('#missing-button').click({ timeout: 2000 });
});

test.describe('checkout', () => {
  test('calls a method that does not exist', async ({ page }) => {
    await page.setContent('<input id="first-name">');
    await (page as any).typeInto('#first-name', 'Ada');
  });

  test('checks the wrong text', async ({ page }) => {
    await page.setContent('<h1 id="title">Products</h1>');
    await expect(page.locator('#title')).toHaveText('wrong', { timeout: 2000 });
  });
});
