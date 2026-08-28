import { Card } from "./ui";

export default function PagePlaceholder({ title, note }: { title: string; note: string }) {
  return (
    <div className="max-w-2xl">
      <h1 className="mb-1 text-2xl font-semibold text-white">{title}</h1>
      <p className="mb-6 text-sm text-zinc-500">{note}</p>
      <Card className="p-8 text-center text-zinc-500">Bu ekran bir sonraki aşamada geliyor 🚧</Card>
    </div>
  );
}
