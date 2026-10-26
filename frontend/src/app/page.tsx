"use client";

import { useEffect, useMemo, useState } from "react";

const MAX_IMAGES = 14;

type Slot = {
  key: string;
  ask: string;
  many: boolean;
  required: boolean;
};

type ChosenFile = {
  file: File;
  url: string;
};

export default function Home() {
  const [slots, setSlots] = useState<Slot[]>([]);
  const [chosen, setChosen] = useState<Record<string, ChosenFile[]>>({});
  const [skipped, setSkipped] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ image: string; summary: string } | null>(null);

  useEffect(() => {
    fetch("/api/slots")
      .then((response) => {
        if (!response.ok) throw new Error("The garment list did not load.");
        return response.json() as Promise<Slot[]>;
      })
      .then((payload) => {
        setSlots(payload);
        setChosen(Object.fromEntries(payload.map((slot) => [slot.key, []])));
      })
      .catch(() => {
        setStatus("The page could not reach the API. Start it with python backend/serve.py.");
      });
  }, []);

  const total = useMemo(
    () => Object.values(chosen).reduce((sum, files) => sum + files.length, 0),
    [chosen],
  );

  function clearSlot(key: string) {
    setChosen((current) => {
      for (const item of current[key] ?? []) URL.revokeObjectURL(item.url);
      return { ...current, [key]: [] };
    });
  }

  function takeFiles(slot: Slot, list: FileList | null) {
    if (!list || skipped[slot.key]) return;
    const existing = chosen[slot.key] ?? [];
    const others = total - existing.length;
    const capacity = MAX_IMAGES - others;
    const allowed = slot.many ? Math.max(0, capacity - existing.length) : Math.max(0, capacity);
    if (allowed <= 0) {
      setStatus(`The limit is ${MAX_IMAGES} photos. Remove one to add another.`);
      return;
    }
    let next = Array.from(list);
    if (!slot.many) next = next.slice(0, 1);
    if (next.length > allowed) {
      next = next.slice(0, allowed);
      setStatus(`Only ${allowed} more photo${allowed === 1 ? "" : "s"} fit in the limit.`);
    } else {
      setStatus("");
    }
    const additions = next.map((file) => ({ file, url: URL.createObjectURL(file) }));
    setChosen((current) => {
      const previous = slot.many ? (current[slot.key] ?? []) : [];
      if (!slot.many) {
        for (const item of current[slot.key] ?? []) URL.revokeObjectURL(item.url);
      }
      return { ...current, [slot.key]: previous.concat(additions) };
    });
  }

  function removeFile(key: string, url: string) {
    setChosen((current) => {
      const item = (current[key] ?? []).find((entry) => entry.url === url);
      if (item) URL.revokeObjectURL(item.url);
      return { ...current, [key]: (current[key] ?? []).filter((entry) => entry.url !== url) };
    });
  }

  async function generate() {
    const missing = slots.find((slot) => {
      if (!slot.required || slot.many || skipped[slot.key]) return false;
      return (chosen[slot.key] ?? []).length === 0;
    });
    if (missing) {
      setStatus(`Add a photo for “${missing.ask}”, or skip that piece.`);
      return;
    }
    if (total === 0) {
      setStatus("Add at least one photo.");
      return;
    }

    setBusy(true);
    setResult(null);
    const session = crypto.randomUUID();
    const queue = slots.flatMap((slot) =>
      (chosen[slot.key] ?? []).map((item) => ({ slot, file: item.file })),
    );
    try {
      for (let index = 0; index < queue.length; index += 1) {
        const { slot, file } = queue[index];
        setStatus(`Uploading ${index + 1} of ${queue.length}…`);
        const response = await fetch("/api/upload", {
          method: "POST",
          headers: {
            "X-Session": session,
            "X-Slot": slot.key,
            "X-Filename": encodeURIComponent(file.name),
          },
          body: file,
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) throw new Error(payload.error || "Upload failed.");
      }
      setStatus("Generating the portrait. A 4K image can take several minutes.");
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session }),
      });
      const payload = (await response.json()) as { error?: string; image?: string; summary?: string };
      if (!response.ok || !payload.image) throw new Error(payload.error || "Generation failed.");
      setResult({ image: payload.image, summary: payload.summary ?? "" });
      setStatus("Done.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-[680px] px-5 pt-10 pb-20">
      <h1 className="text-[28px] font-medium tracking-tight">Silk Studio</h1>
      <p className="mt-2 mb-7 text-[#5c5c5c]">
        Add each piece of one outfit. The portrait appears on this page.
      </p>
      <p className="mb-2 text-[13px] tracking-wide text-[#5c5c5c] uppercase">
        {total} of {MAX_IMAGES} photos
      </p>

      {slots.map((slot) => (
        <section key={slot.key} className="border-t border-[#e4e4e4] py-4">
          <h2 className="text-base font-medium">{slot.ask}</h2>
          <p className="mt-1 mb-3 text-sm text-[#5c5c5c]">
            {slot.many
              ? "None or many. Add a close-up for each detail you need to keep sharp."
              : "One photo. Skip this only when the outfit has no separate picture of this piece."}
          </p>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
            <label
              className={`rounded-full border border-[#1a1a1a] bg-white px-3 py-1.5 text-sm ${
                skipped[slot.key] ? "cursor-default opacity-40" : "cursor-pointer"
              }`}
            >
              {slot.many ? "Add photos" : "Choose photo"}
              <input
                className="hidden"
                type="file"
                accept="image/jpeg,image/png,image/webp,image/gif"
                multiple={slot.many}
                disabled={busy || skipped[slot.key]}
                onChange={(event) => {
                  takeFiles(slot, event.target.files);
                  event.target.value = "";
                }}
              />
            </label>
            {slot.required && !slot.many ? (
              <label className="inline-flex items-center gap-1.5 text-sm text-[#5c5c5c]">
                <input
                  type="checkbox"
                  checked={Boolean(skipped[slot.key])}
                  onChange={(event) => {
                    const next = event.target.checked;
                    setSkipped((current) => ({ ...current, [slot.key]: next }));
                    if (next) clearSlot(slot.key);
                  }}
                />
                Skip this piece
              </label>
            ) : null}
          </div>
          <div className="mt-3 flex flex-wrap gap-2.5">
            {(chosen[slot.key] ?? []).map((item) => (
              <figure key={item.url} className="w-24">
                <img
                  src={item.url}
                  alt=""
                  className="h-24 w-24 rounded-md border border-[#e4e4e4] bg-white object-cover"
                />
                <figcaption className="mt-1 text-xs break-words text-[#5c5c5c]">
                  {item.file.name}
                  <button
                    type="button"
                    className="mt-0.5 block underline"
                    onClick={() => removeFile(slot.key, item.url)}
                  >
                    Remove
                  </button>
                </figcaption>
              </figure>
            ))}
          </div>
        </section>
      ))}

      <div className="mt-6">
        <p className="mb-3 min-h-6 text-[#5c5c5c]" role="status">
          {status}
        </p>
        <button
          type="button"
          disabled={busy}
          onClick={generate}
          className="rounded-full bg-[#1a1a1a] px-4 py-3 text-[15px] text-white disabled:cursor-default disabled:opacity-45"
        >
          Generate portrait
        </button>
      </div>

      {result ? (
        <div className="mt-7">
          <img
            src={result.image}
            alt="Generated fashion portrait"
            className="h-auto w-full rounded-lg bg-white"
          />
          <p className="mt-3">
            <a href={result.image} download className="underline">
              Save image
            </a>
          </p>
          <pre className="mt-3 font-mono text-xs whitespace-pre-wrap text-[#5c5c5c]">
            {result.summary}
          </pre>
        </div>
      ) : null}
    </main>
  );
}
