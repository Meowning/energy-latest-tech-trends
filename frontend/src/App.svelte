<script>
  import { Router, Route, Link } from 'svelte-routing';
  import Home from './routes/Home.svelte';
</script>

<Router>
  <nav>
    <Link to="/">홈</Link>
  </nav>

  <Route path="/">
    <Home />
  </Route>
</Router>
