import * as Switch from "@radix-ui/react-switch";

/** iOS switch: white knob with a soft shadow, system-green track when on, fill-colored when off. */
export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <Switch.Root
      checked={checked}
      onCheckedChange={onChange}
      aria-label={label}
      className="relative h-[31px] w-[51px] shrink-0 rounded-full bg-fill outline-none transition-colors duration-[300ms] data-[state=checked]:bg-green"
    >
      <Switch.Thumb className="block h-[27px] w-[27px] translate-x-[2px] rounded-full bg-thumb shadow-[var(--shadow-thumb)] transition-transform duration-[380ms] ease-[cubic-bezier(0.34,1.56,0.64,1)] data-[state=checked]:translate-x-[22px]" />
    </Switch.Root>
  );
}
