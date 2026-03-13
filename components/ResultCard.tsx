interface Props {
  index: number;
  text: string;
  page: number;
  source: string;
  score: number;
}

function relevanceBadge(score: number): { label: string; className: string } {
  const pct = Math.round(score * 100);
  if (score >= 0.75) {
    return { label: `${pct}% match`, className: "bg-green-50 text-green-700" };
  }
  if (score >= 0.55) {
    return { label: `${pct}% match`, className: "bg-yellow-50 text-yellow-700" };
  }
  return { label: `${pct}% match`, className: "bg-gray-100 text-gray-600" };
}

export default function ResultCard({ index, text, page, source, score }: Props) {
  const badge = relevanceBadge(score);

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm animate-slide-in">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
          Result {index + 1}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Page {page}</span>
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge.className}`}>
            {badge.label}
          </span>
        </div>
      </div>

      <p className="text-gray-800 text-sm leading-relaxed">{text}</p>

      <p className="mt-4 text-xs text-gray-400 truncate">{source}</p>
    </div>
  );
}
