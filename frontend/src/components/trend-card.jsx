import { cn } from "../lib/utils"

function extractHost(url) {
  if (typeof url !== "string") return "Source"
  try {
    return new URL(url).hostname.replace(/^www\./, "")
  } catch {
    return "Source"
  }
}

export function TrendCard({ trend }) {
  if (!trend) return null
  const { title, topic = [], docs = [], img_url: imageUrl } = trend

  return (
    <article className="h-full">
      <div
        className={cn(
          "flex h-full flex-col overflow-hidden rounded-3xl border border-border/60 bg-card/80 text-sm text-muted-foreground shadow-2xl transition-transform duration-300 hover:-translate-y-1 hover:border-border hover:shadow-3xl",
          "card-sheen"
        )}
      >
        <div className="relative h-44 w-full overflow-hidden">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={title}
              className="h-full w-full object-cover"
              loading="lazy"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-white/5 via-transparent to-white/5">
              <span className="text-[0.65rem] uppercase tracking-[0.45em] text-white/40">Visuel</span>
            </div>
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" aria-hidden="true" />
        </div>

        <div className="flex flex-1 flex-col gap-5 p-6">
          <div className="space-y-2">
            <p className="text-base font-semibold leading-tight text-foreground">{title}</p>
            {topic.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {topic.map((label) => (
                  <span
                    key={label}
                    className="rounded-full border border-border/60 bg-white/5 px-3 py-1 text-[0.7rem] uppercase tracking-[0.12em] text-white/70"
                  >
                    {label}
                  </span>
                ))}
              </div>
            )}
          </div>

          {docs.length > 0 && (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {docs.map((doc) => {
                const url = doc?.[2]
                const host = extractHost(url)
                return (
                  <a
                    key={url}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-xs font-medium uppercase tracking-[0.15em] text-white/80 transition hover:border-white/30 hover:bg-white/20"
                  >
                    {host}
                  </a>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </article>
  )
}
