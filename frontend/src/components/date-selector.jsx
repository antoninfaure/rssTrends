import { useMemo, useState } from "react"
import { format, parse, isValid } from "date-fns"
import { fr } from "date-fns/locale"
import { CalendarIcon } from "lucide-react"

import { Button } from "./ui/button"
import { Calendar } from "./ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import { cn } from "../lib/utils"

const DATE_FORMAT = "dd-MM-yyyy"

function parseDateString(value) {
  if (!value) return undefined
  const parsed = parse(value, DATE_FORMAT, new Date())
  return isValid(parsed) ? parsed : undefined
}

export function DateSelector({ dates = [], value, onChange }) {
  const [open, setOpen] = useState(false)

  const selectedDate = useMemo(() => parseDateString(value), [value])

  const availableDates = useMemo(() => {
    return new Set(dates.filter(Boolean))
  }, [dates])

  function handleSelect(date) {
    if (!date) return
    const formatted = format(date, DATE_FORMAT)
    if (!availableDates.has(formatted)) {
      return
    }
    onChange?.(formatted)
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal md:w-64",
            !selectedDate && "text-muted-foreground"
          )}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {selectedDate ? format(selectedDate, "PPP", { locale: fr }) : "Choisir une date"}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={selectedDate}
          onSelect={handleSelect}
          locale={fr}
          initialFocus
          disabled={(date) => !availableDates.has(format(date, DATE_FORMAT))}
          modifiers={{ available: (date) => availableDates.has(format(date, DATE_FORMAT)) }}
          modifiersClassNames={{ available: "font-semibold" }}
        />
      </PopoverContent>
    </Popover>
  )
}
