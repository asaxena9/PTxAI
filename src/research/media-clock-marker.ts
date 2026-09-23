/** Compression-robust visual identity marker for browser clock calibration. */

export const MEDIA_CLOCK_MARKER_SCHEMA =
  "relay.media-clock-visual-frame-identity/1" as const;

export const MEDIA_CLOCK_MARKER_COLUMNS = 24;
export const MEDIA_CLOCK_MARKER_SYNC_BITS = [1, 0, 1, 0] as const;
export const MEDIA_CLOCK_MARKER_ID_BITS = 16;
export const MEDIA_CLOCK_MARKER_CHECKSUM_BITS = 4;

export interface MediaClockMarkerGeometry {
  readonly x: number;
  readonly y: number;
  readonly cellWidth: number;
  readonly cellHeight: number;
}

export interface MediaClockMarkerDecode {
  readonly valid: boolean;
  readonly frameIdentity: number | null;
  readonly confidence: number;
  readonly threshold: number;
  readonly bits: readonly number[];
  readonly reason: string;
}

function checksumNibble(frameIdentity: number): number {
  return (
    0xa ^
    (frameIdentity & 0xf) ^
    ((frameIdentity >>> 4) & 0xf) ^
    ((frameIdentity >>> 8) & 0xf) ^
    ((frameIdentity >>> 12) & 0xf)
  ) & 0xf;
}

function integerBits(value: number, width: number): number[] {
  return Array.from({ length: width }, (_, index) =>
    (value >>> (width - 1 - index)) & 1,
  );
}

function bitsInteger(bits: readonly number[]): number {
  return bits.reduce((value, bit) => (value << 1) | bit, 0) >>> 0;
}

export function encodeMediaClockMarkerBits(frameIdentity: number): number[] {
  if (
    !Number.isSafeInteger(frameIdentity) ||
    frameIdentity < 0 ||
    frameIdentity > 0xffff
  ) {
    throw new Error("Frame identity must be an integer within [0,65535].");
  }
  return [
    ...MEDIA_CLOCK_MARKER_SYNC_BITS,
    ...integerBits(frameIdentity, MEDIA_CLOCK_MARKER_ID_BITS),
    ...integerBits(checksumNibble(frameIdentity), MEDIA_CLOCK_MARKER_CHECKSUM_BITS),
  ];
}

/** Decode normalized cell luminances without repairing uncertain bits. */
export function decodeMediaClockMarkerLuminances(
  luminances: readonly number[],
  threshold = 0.5,
): MediaClockMarkerDecode {
  if (
    luminances.length !== MEDIA_CLOCK_MARKER_COLUMNS ||
    !Number.isFinite(threshold) ||
    threshold <= 0 ||
    threshold >= 1 ||
    luminances.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
  ) {
    return {
      valid: false,
      frameIdentity: null,
      confidence: 0,
      threshold,
      bits: [],
      reason: "Marker luminances or threshold are invalid.",
    };
  }
  const bits = luminances.map((value) => (value >= threshold ? 1 : 0));
  const confidence = Math.min(
    ...luminances.map((value) => Math.min(1, Math.abs(value - threshold) * 2)),
  );
  if (
    MEDIA_CLOCK_MARKER_SYNC_BITS.some((bit, index) => bits[index] !== bit)
  ) {
    return {
      valid: false,
      frameIdentity: null,
      confidence,
      threshold,
      bits,
      reason: "Marker sync pattern does not match.",
    };
  }
  const identityStart = MEDIA_CLOCK_MARKER_SYNC_BITS.length;
  const identityEnd = identityStart + MEDIA_CLOCK_MARKER_ID_BITS;
  const frameIdentity = bitsInteger(bits.slice(identityStart, identityEnd));
  const observedChecksum = bitsInteger(bits.slice(identityEnd));
  if (observedChecksum !== checksumNibble(frameIdentity)) {
    return {
      valid: false,
      frameIdentity: null,
      confidence,
      threshold,
      bits,
      reason: "Marker checksum does not match.",
    };
  }
  return {
    valid: true,
    frameIdentity,
    confidence,
    threshold,
    bits,
    reason: "Marker identity decoded with valid sync and checksum.",
  };
}

export function drawMediaClockMarker(
  context: CanvasRenderingContext2D,
  frameIdentity: number,
  geometry: MediaClockMarkerGeometry,
): void {
  const bits = encodeMediaClockMarkerBits(frameIdentity);
  context.save();
  context.imageSmoothingEnabled = false;
  context.fillStyle = "#7f00ff";
  context.fillRect(
    geometry.x - 4,
    geometry.y - 4,
    geometry.cellWidth * bits.length + 8,
    geometry.cellHeight + 8,
  );
  bits.forEach((bit, index) => {
    context.fillStyle = bit ? "#ffffff" : "#000000";
    context.fillRect(
      geometry.x + index * geometry.cellWidth,
      geometry.y,
      geometry.cellWidth,
      geometry.cellHeight,
    );
  });
  context.restore();
}

export function decodeMediaClockMarkerFromCanvas(
  context: CanvasRenderingContext2D,
  geometry: MediaClockMarkerGeometry,
): MediaClockMarkerDecode {
  const luminances = Array.from(
    { length: MEDIA_CLOCK_MARKER_COLUMNS },
    (_, index) => {
      const sampleX = Math.floor(
        geometry.x + index * geometry.cellWidth + geometry.cellWidth / 2,
      );
      const sampleY = Math.floor(geometry.y + geometry.cellHeight / 2);
      const [red, green, blue] = context.getImageData(sampleX, sampleY, 1, 1).data;
      return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
    },
  );
  return decodeMediaClockMarkerLuminances(luminances);
}
