import { readFileSync } from 'fs';
import { join } from 'path';
import { resolveThemeColors } from './resolve-theme';

// Generate amethyst themes (since amethyst is the default from config)
function generateAmethystThemes() {
  // Read amethyst.json from vendor-src
  const amethystPath = join(import.meta.dir, '../../../vendor-src/oh-my-pi/packages/tui/src/theme/defaults/amethyst.json');
  const amethystContent = readFileSync(amethystPath, 'utf8');
  const amethyst = JSON.parse(amethystContent);
  
  // Create dark and light versions
  const amethystDark = {
    ...amethyst,
    name: 'amethyst-dark',
    appearance: 'dark'
  };
  
  const amethystLight = {
    ...amethyst,
    name: 'amethyst-light',
    appearance: 'light'
  };
  
  return [
    { name: 'amethyst-dark', theme: amethystDark, appearance: 'dark' },
    { name: 'amethyst-light', theme: amethystLight, appearance: 'light' }
  ];
}

function main() {
  try {
    console.log('Resolving theme colors...');
    
    // Generate amethyst themes
    const amethystThemes = generateAmethystThemes();
    
    // Process the copied theme files
    const copiedThemes = [
      { name: 'dark', path: join(import.meta.dir, 'dark.json'), appearance: 'dark' },
      { name: 'light', path: join(import.meta.dir, 'light.json'), appearance: 'light' },
      { name: 'dusk', path: join(import.meta.dir, 'dusk.json'), appearance: 'dark' }
    ];
    
    const allThemes = [...amethystThemes, ...copiedThemes];
    
    // Resolve all themes
    const resolvedThemes = allThemes.map(({ name, path, appearance }) => {
      console.log(`Resolving ${name} (${appearance})...`);
      const themeContent = readFileSync(path, 'utf8');
      const theme = JSON.parse(themeContent);
      
      const resolvedColors = resolveThemeColors(theme.colors, theme.vars || {});
      const resolvedVars = theme.vars ? resolveThemeColors(theme.vars, {}) : {};
      
      // Create the resolved theme object
      const resolvedTheme = {
        name,
        appearance,
        vars: resolvedVars,
        colors: resolvedColors,
        export: theme.export || {},
        terminal: theme.terminal || {}
      };
      
      return resolvedTheme;
    });
    
    // Create expected.json with all resolved themes
    const expected = {
      generatedAt: new Date().toISOString(),
      themes: resolvedThemes,
      schema: {
        version: '1.0',
        description: 'Resolved theme colors for testing',
        tokens: [
          'accent', 'muted', 'dim', 'error', 'warning', 'success', 'thinkingHigh',
          'text', 'thinkingText', 'selectedBg', 'userMessageBg', 'userMessageText',
          'customMessageBg', 'customMessageText', 'customMessageLabel', 
          'toolPendingBg', 'toolSuccessBg', 'toolErrorBg', 'toolOutput',
          'bashMode', 'pythonMode', 'statusLineModel', 'statusLineBg', 
          'statusLineSep', 'statusLinePath', 'statusLineGitClean', 
          'statusLineGitDirty', 'statusLineContext', 'statusLineSpend',
          'statusLineStaged', 'statusLineDirty', 'statusLineUntracked', 
          'statusLineOutput', 'statusLineCost', 'statusLineSubagents',
          'mdCode', 'mdCodeBlock', 'mdQuote', 'mdHeading', 'mdLink', 'mdLinkUrl',
          'mdHr', 'mdListBullet'
        ]
      }
    };
    
    // Write to expected.json
    const fs = require('fs');
    fs.writeFileSync(join(import.meta.dir, 'expected.json'), 
      JSON.stringify(expected, null, 2));
    
    console.log('Theme resolution complete!');
    console.log(`Resolved ${resolvedThemes.length} themes`);
    
  } catch (error) {
    console.error('Error resolving themes:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

export { main, generateAmethystThemes };