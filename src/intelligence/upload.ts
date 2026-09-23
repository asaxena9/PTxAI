"use client";

export interface UploadExtraction {
  text: string;
  method: "plain-text" | "pdf-text" | "image-ocr";
  notice: string;
}

async function extractPdf(file: File): Promise<UploadExtraction> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
  const data = new Uint8Array(await file.arrayBuffer());
  const document = await pdfjs.getDocument({ data }).promise;
  const pages: string[] = [];
  for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
    const page = await document.getPage(pageNumber);
    const content = await page.getTextContent();
    const text = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ");
    pages.push(text);
  }
  return {
    text: pages.join("\n").trim(),
    method: "pdf-text",
    notice: `Read text from ${document.numPages} PDF page${
      document.numPages === 1 ? "" : "s"
    } on this device.`,
  };
}

async function extractImage(file: File): Promise<UploadExtraction> {
  const { createWorker } = await import("tesseract.js");
  const worker = await createWorker("eng");
  try {
    const result = await worker.recognize(file);
    return {
      text: result.data.text.trim(),
      method: "image-ocr",
      notice:
        "Read the photograph on this device. Check every number, side, and restriction before confirming.",
    };
  } finally {
    await worker.terminate();
  }
}

export async function extractInstructionsFromFile(
  file: File,
): Promise<UploadExtraction> {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (file.type === "application/pdf" || extension === "pdf") {
    return extractPdf(file);
  }
  if (file.type.startsWith("image/")) {
    return extractImage(file);
  }
  const text = await file.text();
  return {
    text: text.trim(),
    method: "plain-text",
    notice: "Read the uploaded text file on this device.",
  };
}
