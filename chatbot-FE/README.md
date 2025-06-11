# Uporabniški vmesnik pogovornega robota

Ta direktorij vsebuje kodo uporabniškega vmesnika za pogovornega robota, specializiranega za svetovanje v zvezi z **Evropskim aktom o umetni inteligenci (AI Act)**.
Aplikacija omogoča enostavno in intuitivno interakcijo s sistemom prek spletnega vmesnika.

---

## Zahteve za zagon

* **Node.js**: `v22.4.0`
* **npm**: `v10.8.1`

> **Pomembno:** Pred zagonom je potrebno imeti aktivno verzijo `Node.js 22.4.0` in ustrezen `npm 10.8.1`. Priporočena je uporaba
> orodja `nvm` za lažje upravljanje verzij.

---

## Zagon aplikacije

1. **Namestitev vseh odvisnosti:**

```bash
npm install
```

2. **Zagon aplikacije:**

```bash
npm run dev
```

Po zagonu bo aplikacija dostopna na naslovu `http://localhost:5173/`.

> **Opomba:** Za pravilno delovanje mora biti zagnan tudi **zaledni del** aplikacije (backend). Navodila za njegov zagon
> se nahajajo v mapi [`PROJECT_ROOT/chatbot/`](../chatbot/).
