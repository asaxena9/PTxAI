"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useSpokenCoach() {
  const [muted, setMutedState] = useState(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem("relay.sound-muted.v1") === "true";
    } catch {
      return false;
    }
  });
  const lastSpoken = useRef("");
  const setMuted = useCallback((next: boolean) => {
    setMutedState(next);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem("relay.sound-muted.v1", String(next));
      } catch {
        // Sound remains usable for this session when browser storage is blocked.
      }
    }
  }, []);

  const speak = useCallback(
    (message: string, urgent = false) => {
      if (
        muted ||
        typeof window === "undefined" ||
        !("speechSynthesis" in window) ||
        !message ||
        (!urgent && lastSpoken.current === message)
      ) {
        return;
      }
      if (urgent) window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 0.94;
      utterance.pitch = 1;
      utterance.volume = 1;
      lastSpoken.current = message;
      window.speechSynthesis.speak(utterance);
    },
    [muted],
  );

  useEffect(
    () => () => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    },
    [],
  );

  return { muted, setMuted, speak };
}
