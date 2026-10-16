/**
 * Utility for rendering handwritten document pages to HTML5 Canvas
 * and exporting as high-resolution PNG or PDF.
 */
import { jsPDF } from 'jspdf';

export interface PageConfig {
  pageSize: 'A4' | 'Letter' | 'Note';
  paperStyle: 'lined' | 'plain' | 'grid' | 'sepia';
  inkColor: string;
  handwritingFont: string;
  fontSize: number;
  lineHeight: number;
  humanJitter: number; // 0 to 5
}

export const PAGE_DIMENSIONS = {
  A4: { width: 1200, height: 1697, name: 'A4 (210 × 297 mm)', pdfFormat: 'a4' },
  Letter: { width: 1200, height: 1552, name: 'US Letter (8.5 × 11 in)', pdfFormat: 'letter' },
  Note: { width: 1000, height: 1333, name: 'Compact Notebook (6 × 8 in)', pdfFormat: [1000, 1333] },
};

/**
 * Splits text into pages and lines according to page dimensions and font measurements.
 */
export function paginateText(
  text: string,
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  fontSize: number,
  lineHeight: number,
  paperStyle: string
): string[][] {
  const marginLeft = paperStyle === 'lined' ? 140 : 80;
  const marginRight = 80;
  const marginTop = 120;
  const marginBottom = 100;
  const usableWidth = width - marginLeft - marginRight;
  const usableHeight = height - marginTop - marginBottom;
  const maxLinesPerPage = Math.floor(usableHeight / lineHeight);

  const paragraphs = text.split('\n');
  const pages: string[][] = [[]];
  let currentLines: string[] = pages[0];

  for (const para of paragraphs) {
    if (para.trim().length === 0) {
      if (currentLines.length >= maxLinesPerPage) {
        currentLines = [];
        pages.push(currentLines);
      }
      currentLines.push('');
      continue;
    }

    const words = para.split(' ');
    let currentLine = '';

    for (let w = 0; w < words.length; w++) {
      const word = words[w];
      const testLine = currentLine.length === 0 ? word : `${currentLine} ${word}`;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > usableWidth && currentLine.length > 0) {
        if (currentLines.length >= maxLinesPerPage) {
          currentLines = [];
          pages.push(currentLines);
        }
        currentLines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine.length > 0) {
      if (currentLines.length >= maxLinesPerPage) {
        currentLines = [];
        pages.push(currentLines);
      }
      currentLines.push(currentLine);
    }
  }

  return pages.length > 0 && pages[0].length > 0 ? pages : [['']];
}

/**
 * Draws a single handwritten page onto a canvas context.
 */
export function drawHandwrittenPage(
  ctx: CanvasRenderingContext2D,
  pageLines: string[],
  pageIndex: number,
  totalPages: number,
  config: PageConfig
): void {
  const { width, height } = PAGE_DIMENSIONS[config.pageSize];
  ctx.clearRect(0, 0, width, height);

  // 1. Draw Paper Background
  if (config.paperStyle === 'sepia') {
    ctx.fillStyle = '#f4ecd8';
    ctx.fillRect(0, 0, width, height);
  } else if (config.paperStyle === 'plain') {
    ctx.fillStyle = '#faf8f5';
    ctx.fillRect(0, 0, width, height);
  } else {
    // Lined or Grid
    ctx.fillStyle = '#fbf9f4';
    ctx.fillRect(0, 0, width, height);

    if (config.paperStyle === 'grid') {
      ctx.strokeStyle = 'rgba(203, 213, 225, 0.4)';
      ctx.lineWidth = 1;
      const gridSize = 28;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    } else if (config.paperStyle === 'lined') {
      // Red margin line
      ctx.strokeStyle = '#fca5a5';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(120, 0);
      ctx.lineTo(120, height);
      ctx.stroke();

      // Ruled blue horizontal lines
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      const startY = 120;
      for (let y = startY; y < height - 60; y += config.lineHeight) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }
  }

  // 2. Setup Font & Ink
  ctx.font = `${config.fontSize}px "${config.handwritingFont}", cursive`;
  ctx.fillStyle = config.inkColor;
  ctx.textBaseline = 'alphabetic';

  const marginLeft = config.paperStyle === 'lined' ? 140 : 80;
  let y = 115;

  // 3. Render lines with subtle human jitter
  for (let l = 0; l < pageLines.length; l++) {
    const line = pageLines[l];
    if (line.length > 0) {
      // Add subtle jitter per line
      const jitterY = (Math.sin(l * 1.7) * config.humanJitter);
      const jitterX = (Math.cos(l * 2.3) * (config.humanJitter * 0.7));

      ctx.fillText(line, marginLeft + jitterX, y + jitterY);
    }
    y += config.lineHeight;
  }

  // 4. Page numbering at bottom right
  ctx.font = `14px "JetBrains Mono", monospace`;
  ctx.fillStyle = 'rgba(100, 116, 139, 0.6)';
  ctx.fillText(`Page ${pageIndex + 1} of ${totalPages}`, width - 140, height - 35);
}

/**
 * Exports multiple canvas pages into a single PDF file using jsPDF.
 */
export async function exportToPdf(
  pages: string[][],
  config: PageConfig,
  filename = 'handwritten_document.pdf'
): Promise<void> {
  const { width, height, pdfFormat } = PAGE_DIMENSIONS[config.pageSize];
  
  // Create an offscreen canvas
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const orientation = height >= width ? 'p' : 'l';
  const pdf = new jsPDF({
    orientation,
    unit: 'pt',
    format: pdfFormat as any,
  });

  const pdfWidth = pdf.internal.pageSize.getWidth();
  const pdfHeight = pdf.internal.pageSize.getHeight();

  for (let p = 0; p < pages.length; p++) {
    drawHandwrittenPage(ctx, pages[p], p, pages.length, config);
    const imgData = canvas.toDataURL('image/jpeg', 0.95);

    if (p > 0) {
      pdf.addPage(pdfFormat as any, orientation);
    }
    pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, pdfHeight);
  }

  pdf.save(filename);
}
